#!/usr/bin/env perl
# $Id$
# --
# InjectWorker.pl
# Monitors a directory, and inserts data in the database
# according to entries in the files
# --
# Original script by Olivier Raginel <babar@cern.ch>

use strict;
use warnings;

use Linux::Inotify2;
use POE qw( Wheel::FollowTail Component::Log4perl Wheel::Run Filter::Line );
use POSIX qw( strftime );
use File::Basename;
use DBI;
use Getopt::Long;

# Configuration
my $savedelay      = 300; # Frequency to save offset file, in seconds
my $log4perlConfig = '/opt/injectworker/log4perl.conf';

################################################################################
my $nodbint     = 0; # SM_DONTACCESSDB: no access any DB at all
my $nodbwrite   = 0; # SM_DONTWRITEDB : no write to DB but retrieve HLT key
my $nofilecheck = 0; # SM_NOFILECHECK : no check if files are locally accessible
my $maxhooks    = 3; # Number of parallel hooks
################################################################################

# global vars
chomp( my $host = `hostname -s` );

my $heartbeat = 300;                        # Print a heartbeat every 5 minutes

# get options from environment
# XXX Do we really need to test for existence???
if ( defined $ENV{'SM_NOFILECHECK'} ) {
    $nofilecheck = 1;
}

if ( defined $ENV{'SM_DONTACCESSDB'} ) {
    $nodbint   = 1;
    $nodbwrite = 1;
}

if ( defined $ENV{'SM_DONTWRITEDB'} ) {
    $nodbwrite = 1;
}

# check arguments
unless ( $#ARGV == 2 ) {
    die "Syntax: ./InjectWorker.pl inputpath logpath configfile";
}

my ( $inpath, $logpath, $config ) = @ARGV;
if ( -f $inpath ) {
    die "Error: this version of InjectWorker only supports path.\n"
      . "You might want to simply: cat yourFile > /\$inpath/\$date-manual.log";
}
elsif ( ! -d _ ) {
    die "Error: Specified input path \"$inpath\" does not exist";
}
if ( ! -d $logpath ) {
    die "Error: Specified logpath \"$logpath\" does not exist";
}
if ( ! -e $config ) {
    die "Error: Specified config file \"$config\" does not exist";
}

#########################################################################################

# To rotate logfiles daily
sub get_logfile { return strftime "$logpath/$_[0]-%Y%m%d-$host.log", localtime time; }

# Create logger
Log::Log4perl->init_and_watch( $log4perlConfig, 'HUP');
POE::Component::Log4perl->spawn(
    Alias      => 'logger',
    Category   => 'InjectWorker',
    ConfigFile => $log4perlConfig,
    GetLogfile => \&get_logfile,
);

# Create notifier logger
POE::Component::Log4perl->spawn(
    Alias      => 'notify',
    Category   => 'Notify',
    ConfigFile => $log4perlConfig,
    GetLogfile => \&get_logfile,
);

# Start POE Session, which will do everything
POE::Session->create(
    inline_states => {
        _start       => \&start,
        inotify_poll => sub {
            $_[HEAP]{inotify}->poll;
        },
        watch_hdlr       => \&watch_hdlr,
        save_offsets     => \&save_offsets,
        update_db        => \&update_db,
        read_db_config   => \&read_db_config,
        setup_db         => \&setup_db,
        setup_main_db    => \&setup_main_db,
        setup_hlt_db     => \&setup_hlt_db,
        get_hlt_key      => \&get_hlt_key,
        start_hook       => \&start_hook,
        hook_result      => \&hook_result,
        hook_error       => \&hook_error,
        hook_done        => \&hook_done,
        parse_line       => \&parse_line,
        read_offsets     => \&read_offsets,
        read_changes     => \&read_changes,
        got_log_line     => \&got_log_line,
        got_log_rollover => \&got_log_rollover,
        insert_file      => \&insert_file,
        close_file       => \&close_file,
        shutdown         => \&shutdown,
        heartbeat        => \&heartbeat,
        setup_lock       => \&setup_lock,
        _default         => \&handle_default,
    },
);

POE::Kernel->run();

exit 0;

# Program subs

# time routine for SQL commands timestamp
sub gettimestamp($) {
    return strftime "%Y-%m-%d %H:%M:%S", localtime time;
}

# POE events
sub start {
    my ( $kernel, $heap, $session ) = @_[ KERNEL, HEAP, SESSION ];

    $kernel->yield('setup_db');
    $kernel->yield('setup_lock');
    $kernel->yield('read_offsets');

    # Setup the notify thread
    $kernel->alias_set('notify');
    $heap->{inotify} = new Linux::Inotify2
      or die "Unable to create new inotify object: $!";

    $heap->{inotify}
      ->watch( $inpath, IN_MODIFY, $session->postback("watch_hdlr") )
      or die "Unable to watch dir $inpath: $!";

    open my $inotify_FH, '<&=', $heap->{inotify}->fileno
      or die "Can't fdopen: $!\n";
    $kernel->select_read( $inotify_FH, "inotify_poll" );
    $kernel->alarm( save_offsets => time() + 5 );
    $kernel->call( 'logger', info => "Entering main while loop now" );
}

# lockfile
sub setup_lock {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $lockfile = "/tmp/." . basename($0, '.pl') . ".lock";
    if ( -e $lockfile ) {
        open my $fh, '<', $lockfile
          or die "Error: Lock \"$lockfile\" exists and is unreadable: $!";
        chomp( my $pid = <$fh> );
        close $fh;
        chomp( my $process = `ps -p $pid -o comm=` );
        if( $process && $0 =~ /$process/ ) {
            die "Error: Lock \"$lockfile\" exists, pid $pid (running).";
        }
        elsif( $process ) {
            die "Error: Lock \"$lockfile\" exists, pid $pid (running: $process). Stale lock file?";
        }
        else {
            $kernel->call( 'logger', warn => "Warning: Lock \"$lockfile\""
            . "exists, pid $pid (NOT running). Removing stale lock file?" );
        }
    }
    open my $fh, '>', $lockfile or die "Cannot create $lockfile: $!";
    print $fh "$$\n";    # Fill it with pid
    close $fh;
    $kernel->call( 'logger', info => "Set lock to $lockfile for $$" );

    # Cleanup at the end
    END {
        unlink $lockfile if $lockfile;
    }
    $SIG{TERM} = $SIG{INT} = $SIG{QUIT} = $SIG{HUP} = sub {
        print STDERR "Caught a SIG$_[0] -- Shutting down\n";
        exit 0; # Ensure END block is called
    };
}

# Setup the DB connections, unless NOWRITEDB or NOACCESSDB are set
sub setup_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    # overwrite TNS to be sure it points to new DB
    $ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

    if ($nodbint) {
        $kernel->call( 'logger', warning => "No DB (even!) access flag set" );
    }
    else {
        $kernel->yield( 'read_db_config', $config );
        $kernel->yield('setup_hlt_db');
        if ($nodbwrite) {
            $kernel->call( 'logger',
                warning => "Don't write access DB flag set" );
            $kernel->call( 'logger',
                debug => "Following commands would have been processed:" );
        }
        else {
            $kernel->yield('setup_main_db');
        }
    }
}

sub setup_main_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    $kernel->call( 'logger',
        debug =>
          "Setting up DB connection for $heap->{dbi} and $heap->{reader} write access"
    );
    my $failed = 0;
    $heap->{dbh} = DBI->connect_cached( @$heap{ qw( dbi reader phrase ) } )
      or $failed++;
    if ($failed) {
        $kernel->call( 'logger',
            error => "Connection to $heap->{dbi} failed: $DBI::errstr" );
        $kernel->alarm( setup_main_db => time() + 10 );
        return;
    }

    # Enable DBMS to get statistics
    $heap->{dbh}->func( 1000000, 'dbms_output_enable' );

    my $sql =
        "INSERT INTO CMS_STOMGR.FILES_CREATED ("
      . "FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION,"
      . "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME) "
      . "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $heap->{insertFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "INSERT INTO CMS_STOMGR.FILES_INJECTED ("
      . "FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,INDFILENAME,INDFILESIZE,COMMENT_STR) "
      . "VALUES (?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'),?,?,?)";
    $heap->{closeFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

}

# Setup the HLT DB connection
sub setup_hlt_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    # this is for HLT key queries
    $kernel->call( 'logger',
        debug =>
          "Setting up DB connection for $heap->{hltdbi} and $heap->{hltreader} read access"
    );

    my $failed = 0;
    $heap->{dbhlt} =
      DBI->connect_cached( @$heap{ map { "hlt$_" } qw( dbi reader phrase ) } )
      or $failed++;
    if ($failed) {
        $kernel->call( 'logger',
            error => "Connection to $heap->{hltdbi} failed: $DBI::errstr" );
        $kernel->alarm( setup_hlt_db => time() + 10 );
        return;
    }

    my $sql = "SELECT STRING_VALUE FROM CMS_RUNINFO.RUNSESSION_PARAMETER "
      . "WHERE RUNNUMBER=? and NAME='CMS.LVL0:HLT_KEY_DESCRIPTION'";
    $heap->{hltHandle} = $heap->{dbhlt}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbhlt}->errstr;
}

# Retrieve the HLT key for a given run
sub get_hlt_key {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    my $runnumber = $heap->{args}->{RUNNUMBER};
    unless ( defined $runnumber ) {
        $kernel->call( 'logger', error => "Trying to get HLT key without RUNNUMBER!" );
        return;
    }
    my $hltkey = $heap->{hltkeys}->{$runnumber};

    # query hlt db if hlt was not already obtained
    unless ( defined $hltkey and $heap->{dbhlt} ) {
        my $errflag = 0;
        $kernel->call( 'logger',
            debug => "Quering DB for runnumber $runnumber" );
        $heap->{hltHandle}->execute($runnumber) or $errflag = 1;
        if ($errflag) {
            $kernel->call( 'logger',
                error => "DB query get HLT KEY for run $runnumber returned "
                . $heap->{hltHandle}->errstr );
        }
        else {
            ($heap->{hltkeys}->{$runnumber}) = ($hltkey) =
                $heap->{hltHandle}->fetchrow_array
            or $errflag = 1;
            if ($errflag) {
                $kernel->call( 'logger',
                    error => "Fetching HLT KEY for run $runnumber returned "
                    . $heap->{hltHandle}->errstr );
            }
            else {
                $kernel->call( 'logger',
                    debug => "Obtained HLT key $hltkey for run $runnumber" );
            }
        }
        $heap->{hltHandle}->finish;
    }
    unless( defined $hltkey ) {
        $kernel->call( 'logger', error => "Could not get an HLTKEY for run
            $runnumber" );
    }
    else {
        $heap->{args}->{COMMENT} = 'HLTKEY=' . $hltkey;
        $heap->{args}->{HLTKEY}  = $hltkey;
    }
}

# Parse lines like
#./closeFile.pl  --FILENAME Data.00133697.0135.Express.storageManager.00.0000.dat --FILECOUNTER 0 --NEVENTS 21 --FILESIZE 1508412 --STARTTIME 1271857503 --STOPTIME 1271857518 --STATUS closed --RUNNUMBER 133697 --LUMISECTION 135 --PATHNAME /store/global//02/closed/ --HOSTNAME srv-C2C06-12 --SETUPLABEL Data --STREAM Express --INSTANCE 0 --SAFETY 0 --APPVERSION CMSSW_3_5_4_onlpatch3_ONLINE --APPNAME CMSSW --TYPE streamer --DEBUGCLOSE 1 --CHECKSUM c8b5a624 --CHECKSUMIND 8912d364
sub parse_line {
    my ( $kernel, $heap, $line ) = @_[ KERNEL, HEAP, ARG0 ];
    my @args = split / +/, $line;
    shift @args;    # remove ./closeFile.pl or ./insertFile.pl
    my %args;
    if ( @args % 2 ) {
        $kernel->call( 'logger', error => "Could not parse line $line!" );
    }
    else {

        # Even number of arguments, processing
        delete $heap->{args};
        for ( my $i = 0 ; $i < $#args ; $i += 2 ) {
            my $value = $args[ $i + 1 ];
            for( $args[$i] ) {
                s/^--//;
#                s/_//g; # This breaks too many things
                s/^COUNT$/FILECOUNTER/;
                s/^DATASET$/SETUPLABEL/;
                $heap->{args}->{$_} = $value;
                $heap->{args}->{$_} = $value if s/_//g;
            }
        }
    }
}

# injection subroutine
sub update_db {
    my ( $kernel, $heap, $handler, @params ) = @_[ KERNEL, HEAP, ARG0 .. $#_ ];

    # Check that all required parameters are there
    my @bind_params;
    for ( @params) {
        if ( exists $heap->{args}->{$_} ) {
            my $value = $heap->{args}->{$_};
            if (/TIME$/) {
                $value = gettimestamp $value;
            }
            push @bind_params, $value;
        }
        else {
            $kernel->call( 'logger', error => "$handler failed: Could not obtain parameter $_" );
            return;
        }
    }
    # redirect setuplabel/streams to different destinations according to
    # https://twiki.cern.ch/twiki/bin/view/CMS/SMT0StreamTransferOptions
    my $stream = $heap->{args}->{STREAM};
    return
      if ( $stream eq 'EcalCalibration'
        || $stream =~ '_EcalNFS$' )    #skip EcalCalibration
      || $stream =~ '_NoTransfer$';    #skip if NoTransfer option is set

    my $errflag = 0;
    my $rows = $heap->{$handler}->execute( @bind_params ) or $errflag = 1;

    $rows = 'undef' unless defined $rows;
    if ($errflag) {
        $kernel->call( 'logger',
            error => "DB access error (rows: $rows)"
              . " for $handler when executing ("
              . join( ', ', @bind_params ) . '), DB returned: '
              . $heap->{$handler}->errstr );
        return;
    }

    # Print any messages from the DB's dbms output
    for ( $heap->{dbh}->func('dbms_output_get') ) {
        $kernel->call( 'logger', info => $_ );
    }

    if ( $rows != 1 ) {
        $kernel->call( 'logger',
            error => "DB did not return one row for $handler ("
            . join( ', ', @bind_params ) . "), but $rows" );
        return;
    }
}

sub insert_file {
    my ( $kernel, $session, $heap ) = @_[ KERNEL, SESSION, HEAP ];

    $heap->{args}->{PRODUCER} = 'StorageManager';
    $kernel->call( $session, 'update_db', insertFile => qw(
        FILENAME PATHNAME HOSTNAME SETUPLABEL STREAM TYPE
        PRODUCER APPNAME APPVERSION RUNNUMBER LUMISECTION
        FILECOUNTER INSTANCE STARTTIME
        ) );
}

sub close_file {
    my ( $kernel, $session, $heap ) = @_[ KERNEL, SESSION, HEAP ];

    # index file name and size
    $heap->{args}->{INDFILE} = $heap->{args}->{FILENAME};
    $heap->{args}->{INDFILE} =~ s/\.dat$/\.ind/;
    $heap->{args}->{INDFILESIZE} = -1;
    $kernel->call( 'logger', debug => "Indfile: $heap->{args}->{PATHNAME}/$heap->{args}->{INDFILE} for $heap->{args}->{FILENAME}" );
    if ( $host eq $heap->{args}->{HOSTNAME} ) {
        if ( -e "$heap->{args}->{PATHNAME}/$heap->{args}->{INDFILE}" ) {
            $heap->{args}->{INDFILESIZE} =  (stat(_))[7];
        }
        elsif ( $nofilecheck == 0 ) {
            $heap->{args}->{INDFILE} = '';
        }
    }

    $heap->{args}->{DESTINATION} = 'Global';
    my $setuplabel               = $heap->{args}->{SETUPLABEL};
    my $stream                   = $heap->{args}->{STREAM};
    if (   $setuplabel =~ 'TransferTest'
        || $stream =~ '_TransferTest$' )
    {
        $heap->{args}->{DESTINATION} =
          'TransferTest';    # transfer but delete after
    }
    elsif ( $stream =~ '_NoRepack$' || $stream eq 'Error' ) {
        $heap->{args}->{DESTINATION} = 'GlobalNoRepacking';    # do not repack
        $heap->{args}->{INDFILE}     = '';
        $heap->{args}->{INDFILESIZE} = -1;
    }

    $kernel->call( $session, 'update_db', closeFile => qw(
        FILENAME PATHNAME DESTINATION NEVENTS FILESIZE
        CHECKSUM STOPTIME INDFILE INDFILESIZE COMMENT
        ) );

    # Alias index for Tier0
    $heap->{args}->{INDEX} = $heap->{args}->{INDFILE};

    # Run the hook
    $kernel->call( $session, 'start_hook' );

    # XXX Write a proper log for the notification part
    # XXX Remove this duplicate code...
    # redirect setuplabel/streams to different destinations according to
    # https://twiki.cern.ch/twiki/bin/view/CMS/SMT0StreamTransferOptions
    return
      if $stream eq 'EcalCalibration'
        || $stream =~ '_EcalNFS$'     #skip EcalCalibration
        || $stream =~ '_NoTransfer$'    #skip if NoTransfer option is set
        || $stream =~ '_DontNotifyT0$'; #skip if DontNotify

    $kernel->post( 'notify', info => join( ' ', 'notifyTier0.pl', grep defined, map {
        exists $heap->{args}->{$_} ? "--$_=$heap->{args}->{$_}" : undef } 
                        qw( APPNAME APPVERSION RUNNUMBER LUMISECTION FILENAME
                        PATHNAME HOSTNAME DESTINATION SETUPLABEL
                        STREAM TYPE NEVENTS FILESIZE CHECKSUM
                        HLTKEY INDEX STARTTIME STOPTIME ) ) );

}

sub start_hook {
    my ( $kernel, $session, $heap ) = @_[ KERNEL, SESSION, HEAP ];
    my $cmd = $ENV{'SM_HOOKSCRIPT'};
    return unless $cmd;
    my @args = grep defined, map {
        exists $heap->{args}->{$_} ? "--$_=$heap->{args}->{$_}" : undef } 
                        qw( APPNAME APPVERSION RUNNUMBER LUMISECTION FILENAME
                        PATHNAME HOSTNAME DESTINATION SETUPLABEL
                        STREAM TYPE NEVENTS FILESIZE CHECKSUM INSTANCE
                        HLTKEY INDEX STARTTIME STOPTIME FILECOUNTER );
    unshift @args, $cmd;
    $kernel->call( 'logger', debug => "Running hook: " . join( " ", @args) );
    my $task = POE::Wheel::Run->new(
            Program      => sub { system( @args ); },
            StdoutFilter => POE::Filter::Line->new(),
            StderrFilter => POE::Filter::Line->new(),
            StdoutEvent  => 'hook_result',
            StderrEvent  => 'hook_error',
            CloseEvent   => 'hook_done',
    );
    $heap->{task}->{$task->ID} = $task;
    $kernel->sig_child($task->PID, "sig_child");
}

# Catch and display information from the hook's STDOUT.
sub hook_result {
    my ($kernel, $result) = @_[ KERNEL, ARG0 ];
    $kernel->call( 'logger', debug => "Hook output: $result" );
}

# Catch and display information from the hook's STDERR.
sub hook_error {
    my ($kernel, $result) = @_[ KERNEL, ARG0 ];
    $kernel->call( 'logger', info => "Hook error: $result" );
}

# The task is done.  Delete the child wheel, and try to start a new
# task to take its place.
sub hook_done {
    my ($kernel, $heap, $task_id) = @_[KERNEL, HEAP, ARG0];
    delete $heap->{task}->{$task_id};
#    $kernel->yield("next_hook"); # For when it will be more clever
}

# Detect the CHLD signal as each of our children exits.
sub sig_child {
    my ($heap, $sig, $pid, $exit_val) = @_[HEAP, ARG0, ARG1, ARG2];
    my $details = delete $heap->{$pid};
}

sub got_log_line {
    my ( $kernel, $session, $heap, $line, $wheelID ) = @_[ KERNEL, SESSION, HEAP, ARG0, ARG1 ];
    my $file = $heap->{watchlist}{$wheelID};
    $kernel->call( 'logger', debug => "In $file, got line: $line" );
    if ( $line =~ /(?:(insert|close)File)/i ) {
        $kernel->call( $session, parse_line => $line );
        $kernel->call( $session, 'get_hlt_key' );
        $kernel->call( 'logger', debug => "Yielding ${1}_file" );
        $kernel->call( $session, "${1}_file" );
    }
}

sub got_log_rollover {
    my ( $kernel, $heap, $wheelID ) = @_[ KERNEL, HEAP, ARG0 ];
    my $file = $heap->{watchlist}{$wheelID};
    $kernel->call( 'logger', info => "$file rolled over" );
}

# Create a watcher for a file, if none exist already
sub read_changes {
    my ( $kernel, $heap, $file ) = @_[ KERNEL, HEAP, ARG0 ];

    return if $heap->{watchlist}{$file};    # File is already monitored
    my $seek = $heap->{offsets}{$file} || 0;
    my $size = ( stat $file )[7];
    if ( $seek > $size ) {
        $kernel->call( 'logger',
            warning =>
"Saved seek ($seek) is greater than current filesize ($size) for $file"
        );
        $seek = 0;
    }
    $kernel->call( 'logger', info => "Watching $file, starting at $seek" );
    my $wheel = POE::Wheel::FollowTail->new(
        Filename   => $file,
        InputEvent => "got_log_line",
        ResetEvent => "got_log_rollover",
        Seek       => $seek,
    );
    $heap->{watchlist}{ $wheel->ID } = $file;
    $heap->{watchlist}{$file} = $wheel;
}

# Clean shutdown
sub shutdown {
    my ( $kernel, $heap, $session ) = @_[ KERNEL, HEAP, SESSION ];
    for ( qw( insertFile closeFile hltHandle ) ) {
        $heap->{$_}->finish;
    }
    for ( dbh dbhlt ) {
        $kernel->call( 'logger', info => "Disconnecting $_" );
        $heap->{$_}->disconnect
          or $kernel->call( 'logger',
            warning => "Disconnection from Oracle $_ failed: $DBI::errstr" );
    }
    die "Shutting down!";
}

# postback called when some iNotify event is raised
# XXX Filter files?
sub watch_hdlr {
    my $kernel = $_[KERNEL];
    my $event  = $_[ARG1][0];
    my $name   = $event->fullname;

    if ( $event->IN_MODIFY ) {
        $kernel->call( 'logger', debug => "$name was modified\n" );
        $kernel->yield( read_changes => $name );
    }
    else {
        $kernel->call( 'logger', warning => "$name is no longer mounted" )
          if $event->IN_UNMOUNT;
        $kernel->call( 'logger', error => "events for $name have been lost" )
          if $event->IN_Q_OVERFLOW;
    }
}

# Read DB configuration properly
sub read_db_config {
    my ( $kernel, $heap, $config ) = @_[ KERNEL, HEAP, ARG0 ];
    if ( $config && -r $config ) {
        open my $fh, '<', $config or die "Cannot open $config: $!";
        while (<$fh>) {
            next if /^\s*#/;
            if ( my ( $key, $value ) = /^\s*\$(\w+)\s*=\s*"?(\S+?)"?\s*;/ ) {
                if( $key =~ /^(?:hlt)?(?:dbi|reader|phrase)$/ ) {
                    $heap->{$key} = $value;
                    next;
                }
                #$heap->{$key} = $value and next if $key =~ /^(?:hlt)?(?:dbi|reader|phrase)$/;
                $kernel->call( 'logger',
                    warning =>
                      "Ignoring unknown configuration variable: $key" );
            }
        }
        close $fh;
    }
    unless ( $heap->{reader} && $heap->{phrase} ) {
        die "No DB configuration. Aborting.";
    }
    $kernel->call( 'logger', info => "Read DB configuration from $config" );
}

# Save the offset for each file so processing can be resumed at any time
sub save_offsets {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    $kernel->call( 'logger', debug => "Saving offsets" );
    $kernel->alarm( save_offsets => time() + $savedelay );
    for my $tailor ( grep { /^[0-9]+$/ } keys %{ $heap->{watchlist} } ) {
        my $file   = $heap->{watchlist}{$tailor};
        my $wheel  = $heap->{watchlist}{$file};
        my $offset = $wheel->tell;
        $heap->{offsets}{$file} = $offset;
    }
    my $savefile = $logpath . '/offset.txt';
    open my $save, '>', $savefile or die "Can't open $savefile: $!";
    while ( my ( $file, $offset ) = each %{ $heap->{offsets} } ) {
        $kernel->call( 'logger',
            debug => "Saving session information for $file: $offset" );
        print $save "$file $offset\n";
    }
    close $save;
}

# Read offsets, that is set the offset for each file to continue
# processing
sub read_offsets {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $savefile = $logpath . '/offset.txt';
    return unless -r $savefile;
    $kernel->call( 'logger', debug => "Reading offset file $savefile..." );
    open my $save, '<', $savefile or die "Can't open $savefile: $!";
    while (<$save>) {
        my ( $file, $offset ) = /^(\S+) ([0-9]+)$/;
        if( -f $file ) {
            my $fsize = (stat(_))[7];
            if( $offset != $fsize ) {
                $kernel->call( 'logger',
                    debug => "File $file has a different size: $offset != $fsize" );
                $kernel->yield( read_changes => $file );
            }
            $heap->{offsets}{$file} = $offset;
            $kernel->call( 'logger',
                debug => "Setting session information for $file: $offset" );
        }
        else {
            $kernel->call( 'logger',
                debug => "Discarding session information for non-existing $file: $offset" );
        }
    }
    close $save;
}

# Print some heartbeat at fixed interval
sub heartbeat {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    $kernel->call( 'logger', info => "Still alive in main loop" );
    $kernel->alarm( heartbeat => time() + $heartbeat );
}

# Do something with all POE events which are not caught
sub handle_default {
    my ( $kernel, $event, $args ) = @_[ KERNEL, ARG0, ARG1 ];
    print STDERR "WARNING: Session ".$_[SESSION]->ID."caught unhandled event $event with (@$args).";
    $kernel->call( 'logger',
            warning => "Session "
          . $_[SESSION]->ID
          . " caught unhandled event $event with (@$args)." );
}
