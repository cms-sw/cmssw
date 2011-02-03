#!/usr/bin/env perl
# $Id: InjectWorker.pl,v 1.57 2010/11/17 11:16:23 babar Exp $
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

################################################################################
my $nodbint     = 0; # SM_DONTACCESSDB: no access any DB at all
my $nodbwrite   = 0; # SM_DONTWRITEDB : no write to DB but retrieve HLT key
my $nofilecheck = 0; # SM_NOFILECHECK : no check if files are locally accessible
my $maxhooks    = 3; # Number of parallel hooks
################################################################################

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
elsif ( !-d _ ) {
    die "Error: Specified input path \"$inpath\" does not exist";
}
if ( !-d $logpath ) {
    die "Error: Specified logpath \"$logpath\" does not exist";
}
if ( !-e $config ) {
    die "Error: Specified config file \"$config\" does not exist";
}

#########################################################################################
# Configuration
chomp( my $host = `hostname -s` );
my $offsetfile = $logpath . '/offset.txt';
my $heartbeat  = 300;                        # Print a heartbeat every 5 minutes
my $savedelay = 300;    # Frequency to save offset file, in seconds
my $log4perlConfig = '/opt/injectworker/log4perl.conf';

# To rotate logfiles daily
sub get_logfile {
    return strftime "$logpath/$_[0]-%Y%m%d-$host.log", localtime time;
}

# Create logger
Log::Log4perl->init_and_watch( $log4perlConfig, 'HUP' );
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
        setup_runcond_db => \&setup_runcond_db,
        get_num_sm       => \&get_num_sm,
        get_hlt_key      => \&get_hlt_key,
        get_from_runcond => \&get_from_runcond,
        start_hook       => \&start_hook,
        next_hook        => \&next_hook,
        hook_result      => \&hook_result,
        hook_error       => \&hook_error,
        hook_done        => \&hook_done,
        parse_line       => \&parse_line,
        parse_lumi_line  => \&parse_lumi_line,
        got_begin_of_run => \&got_begin_of_run,
        got_end_of_run   => \&got_end_of_run,
        got_end_of_lumi  => \&got_end_of_lumi,
        read_offsets     => \&read_offsets,
        read_changes     => \&read_changes,
        got_log_line     => \&got_log_line,
        got_log_rollover => \&got_log_rollover,
        insert_file      => \&insert_file,
        close_file       => \&close_file,
        shutdown         => \&shutdown,
        heartbeat        => \&heartbeat,
        setup_lock       => \&setup_lock,
        sig_child        => \&sig_child,
        sig_abort        => \&sig_abort,
        _stop            => \&shutdown,
        _default         => \&handle_default,
    },

    #    options  => { trace => 1 },
);

POE::Kernel->run();

exit 0;

# Program subs

# time routine for SQL commands timestamp
sub gettimestamp($) {
    return strftime "%Y-%m-%d %H:%M:%S", localtime $_[0];
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

    # Save offset files regularly
    $kernel->delay( save_offsets => $savedelay );

    # Print a heartbeat regularly
    $kernel->delay( heartbeat => $heartbeat );

    $kernel->post( 'logger', info => "Entering main while loop now" );
}

# lockfile
sub setup_lock {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $lockfile = "/tmp/." . basename( $0, '.pl' ) . ".lock";
    if ( -e $lockfile ) {
        open my $fh, '<', $lockfile
          or die "Error: Lock \"$lockfile\" exists and is unreadable: $!";
        chomp( my $pid = <$fh> );
        close $fh;
        chomp( my $process = `ps -p $pid -o comm=` );
        if ( $process && $0 =~ /$process/ ) {
            die "Error: Lock \"$lockfile\" exists, pid $pid (running).";
        }
        elsif ($process) {
            die
"Error: Lock \"$lockfile\" exists, pid $pid (running: $process). Stale lock file?";
        }
        else {
            $kernel->post( 'logger',
                warn => "Warning: Lock \"$lockfile\""
                  . "exists, pid $pid (NOT running). Removing stale lock file?"
            );
        }
    }
    open my $fh, '>', $lockfile or die "Cannot create $lockfile: $!";
    print $fh "$$\n";    # Fill it with pid
    close $fh;
    $kernel->post( 'logger', info => "Set lock to $lockfile for $$" );

    # Cleanup at the end
    END {
        unlink $lockfile if $lockfile;
    }
    $kernel->sig( INT  => 'sig_abort' );
    $kernel->sig( TERM => 'sig_abort' );
    $kernel->sig( QUIT => 'sig_abort' );
}

# Setup the DB connections, unless NOWRITEDB or NOACCESSDB are set
sub setup_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    # overwrite TNS to be sure it points to new DB
    $ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

    if ($nodbint) {
        $kernel->post( 'logger', warning => "No DB (even!) access flag set" );
    }
    else {
        $kernel->yield( 'read_db_config', $config );
        $kernel->yield('setup_runcond_db');
        if ($nodbwrite) {

            # XXX This will not work as there is no further code for that case
            $kernel->post( 'logger',
                warning => "Don't write access DB flag set" );
            $kernel->post( 'logger',
                debug => "Following commands would have been processed:" );
        }
        else {
            $kernel->yield('setup_main_db');
        }
    }
}

sub setup_main_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    $kernel->post( 'logger',
        debug => "Setting up DB connection for $heap->{dbi}"
          . " and $heap->{reader} write access" );
    my $failed = 0;
    $heap->{dbh} = DBI->connect_cached( @$heap{qw( dbi reader phrase )} )
      or $failed++;
    if ($failed) {
        $kernel->post( 'logger',
            error => "Connection to $heap->{dbi} failed: $DBI::errstr" );
        $kernel->delay( setup_main_db => 10 );
        return;
    }

    # Enable DBMS to get statistics
    $heap->{dbh}->func( 1000000, 'dbms_output_enable' );
    my $sths = $heap->{sths};

    # For new files
    my $sql =
        "INSERT INTO CMS_STOMGR.FILES_CREATED "
      . "(FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION,"
      . "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME) "
      . "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $sths->{insertFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For new files: update SM_SUMMARY
    $sql = "BEGIN CMS_STOMGR.FILES_CREATED_PROC_SUMMARY( ? ); END;";
    $sths->{insertFileSummaryProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For new files: update SM_INSTANCES
    $sql = "BEGIN CMS_STOMGR.FILES_CREATED_PROC_INSTANCES( ? ); END;";
    $sths->{insertFileInstancesProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files
    $sql =
        "INSERT INTO CMS_STOMGR.FILES_INJECTED "
      . "(FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,INDFILENAME,INDFILESIZE,COMMENT_STR) "
      . "VALUES (?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'),?,?,?)";
    $sths->{closeFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files: update SM_SUMMARY
    $sql = "BEGIN CMS_STOMGR.FILES_INJECTED_PROC_SUMMARY( ? ); END;";
    $sths->{closeFileSummaryProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files: update SM_INSTANCES
    $sql = "BEGIN CMS_STOMGR.FILES_INJECTED_PROC_INSTANCES( ? ); END;";
    $sths->{closeFileInstancesProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

  # For explanation, see:
  # https://twiki.cern.ch/twiki/bin/viewauth/CMS/StorageManagerEndOfLumiHandling
    $sql =
        "insert into CMS_STOMGR.RUNS "
      . "(RUNNUMBER, INSTANCE, HOSTNAME, N_INSTANCES, N_LUMISECTIONS, "
      . " STATUS, START_TIME)"
      . " values "
      . "(        ?,        ?,        ?,           ?,              0, "
      . "      1, TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $sths->{beginOfRun} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "update CMS_STOMGR.RUNS set "
      . "STATUS = 0, N_LUMISECTIONS = ?, "
      . "END_TIME = TO_DATE(?,'YYYY-MM-DD HH24:MI:SS') "
      . "where RUNNUMBER = ? and INSTANCE = ?";
    $sths->{endOfRun} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "insert into CMS_STOMGR.STREAMS "
      . "(RUNNUMBER, LUMISECTION, STREAM, INSTANCE,"
      . " CTIME,                      FILECOUNT)"
      . " values "
      . "(        ?,           ?,      ?,        ?,"
      . " TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'), ?)";
    $sths->{endOfLumi} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

}

# Setup the run conditions DB connection
sub setup_runcond_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    # this is for HLT key and number of SM queries
    $kernel->post( 'logger',
        debug => "Setting up DB connection for $heap->{dbi}"
          . " and $heap->{reader} read access" );

    my $failed = 0;
    unless ( $heap->{dbh} ) {
        $heap->{dbh} = DBI->connect_cached( @$heap{qw( dbi reader phrase )} )
          or $failed++;
    }
    if ($failed) {
        $kernel->post( 'logger',
            error => "Connection to $heap->{dbi} failed: $DBI::errstr" );
        $kernel->delay( setup_runcond_db => 10 );
        return;
    }

    my $sql = "SELECT STRING_VALUE FROM CMS_RUNINFO.RUNSESSION_PARAMETER "
      . "WHERE RUNNUMBER=? and NAME=?";
    $heap->{sths}->{runCond} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;
}

# Rertieve the number of storage managers from the run conditions DB
sub get_num_sm {
    my ( $kernel, $heap, $callback, $args ) = @_[ KERNEL, HEAP, ARG0 .. ARG2 ];
    $kernel->yield(
        get_from_runcond => 'NumSM',
        'CMS.DAQ:NB_ACTIVE_STORAGEMANAGERS_T', $args
    );
    $kernel->yield( $callback => $args );
}

# Retrieve the HLT key for a given run
sub get_hlt_key {
    my ( $kernel, $heap, $kind, $args ) = @_[ KERNEL, HEAP, ARG0 .. ARG2 ];
    $kernel->yield(
        get_from_runcond => 'HLTkey',
        'CMS.LVL0:HLT_KEY_DESCRIPTION', $args
    );
    $kernel->yield( "${kind}_file" => $args );
}

sub get_from_runcond {
    my ( $kernel, $heap, $kind, $key, $args ) =
      @_[ KERNEL, HEAP, ARG0 .. ARG2 ];

    my $runnumber = $args->{RUNNUMBER} || $args->{run};
    unless ( defined $runnumber and $kind and $key ) {
        $kind      = 'Undefined kind of query' unless $kind;
        $key       = 'Undefined key'           unless $key;
        $runnumber = 'Undefined runnumber'     unless defined $runnumber;
        $kernel->post( 'logger',
            error =>
              "Trying to get $kind (key: $key) for runnumber $runnumber!" );
        return;
    }
    my $cached = $heap->{$kind}->{$runnumber};    # Try to get it from the cache

    # query run conditions db if key was not already obtained for this run
    my $sth = $heap->{sths}->{runCond};
    unless ( defined $cached and $sth ) {
        my $errflag = 0;
        $kernel->post( 'logger',
            debug => "Querying DB for runnumber $runnumber" );
        $sth->execute( $runnumber, $key ) or $errflag = 1;
        if ($errflag) {
            $kernel->post( 'logger',
                error =>
                  "DB query get $kind (key: $key) for run $runnumber returned "
                  . $sth->errstr );
        }
        else {
            ( $heap->{$kind}->{$runnumber} ) = ($cached) = $sth->fetchrow_array
              or $errflag = 1;
            if ($errflag) {
                $kernel->post( 'logger',
                    error =>
                      "Fetching $kind (key: $key) for run $runnumber returned "
                      . $sth->errstr );
            }
            else {
                $kernel->post( 'logger',
                    debug =>
                      "Obtained $kind key $key = $cached for run $runnumber" );
            }
        }
        $sth->finish;
    }
    unless ( defined $cached ) {
        $kernel->post( 'logger',
            error =>
              "Could not retrieve $kind (key: $key) for run $runnumber" );
    }
    elsif ( $kind eq 'HLTkey' ) {
        $args->{COMMENT} = 'HLTKEY=' . $cached;
        $args->{HLTKEY}  = $cached;
    }
    else {
        $args->{n_instances} = $cached;
    }
}

# Parse lines like
#./closeFile.pl  --FILENAME Data.00133697.0135.Express.storageManager.00.0000.dat --FILECOUNTER 0 --NEVENTS 21 --FILESIZE 1508412 --STARTTIME 1271857503 --STOPTIME 1271857518 --STATUS closed --RUNNUMBER 133697 --LUMISECTION 135 --PATHNAME /store/global//02/closed/ --HOSTNAME srv-C2C06-12 --SETUPLABEL Data --STREAM Express --INSTANCE 0 --SAFETY 0 --APPVERSION CMSSW_3_5_4_onlpatch3_ONLINE --APPNAME CMSSW --TYPE streamer --DEBUGCLOSE 1 --CHECKSUM c8b5a624 --CHECKSUMIND 8912d364
sub parse_line {
    my ( $kernel, $heap, $callback, $line ) = @_[ KERNEL, HEAP, ARG0, ARG1 ];
    my @args = split / +/, $line;
    my $kind = shift @args;

    # Test input parameters and return if something fishy is found
    return unless $kind =~ /^\.\/${callback}File\.pl$/;
    if ( @args % 2 ) {
        $kernel->post( 'logger', error => "Could not parse line $line!" );
        return;
    }

    # Even number of arguments, processing
    my %args;
    for ( my $i = 0 ; $i < $#args ; $i += 2 ) {
        my $value = $args[ $i + 1 ];
        for ( $args[$i] ) {
            s/^--//;
            s/^COUNT$/FILECOUNTER/;
            s/^DATASET$/SETUPLABEL/;
            $args{$_} = $value;
            $args{$_} = $value if s/_//g;
        }
    }
    $kernel->yield( get_hlt_key => $callback, \%args );
}

# injection subroutine
sub update_db {
    my ( $kernel, $heap, $args, $handler, @params ) =
      @_[ KERNEL, HEAP, ARG0 .. $#_ ];

    # Check that all required parameters are there
    my @bind_params;
    for (@params) {
        if ( exists $args->{$_} ) {
            my $value = $args->{$_};
            if (/TIME$/) {
                $value = gettimestamp $value;
            }
            push @bind_params, $value;
        }
        else {
            $kernel->post( 'logger',
                error => "$handler failed: Could not obtain parameter $_" );
            return;
        }
    }
    $kernel->post( 'logger',
        debug => "Updating $handler with:"
          . join( " ", map { "$params[$_]=$bind_params[$_]" } 0 .. $#params ) );

    # redirect setuplabel/streams to different destinations according to
    # https://twiki.cern.ch/twiki/bin/view/CMS/SMT0StreamTransferOptions
    my $stream = $args->{STREAM};
    return
      if defined $stream
          and (
                 $stream eq 'EcalCalibration'
              or $stream =~ /_EcalNFS$/      #skip EcalCalibration
              or $stream =~ /_NoTransfer$/
          );                                 #skip if NoTransfer option is set

    my $errflag = 0;
    my $rows = $heap->{sths}->{$handler}->execute(@bind_params) or $errflag = 1;

    $rows = 'undef' unless defined $rows;
    if ($errflag) {
        $kernel->post( 'logger',
                error => "DB access error (rows: $rows)"
              . " for $handler when executing ("
              . join( ', ', @bind_params )
              . '), DB returned: '
              . $heap->{sths}->{$handler}->errstr );
        return;
    }

    # Print any messages from the DB's dbms output
    for ( $heap->{dbh}->func('dbms_output_get') ) {
        $kernel->post( 'logger', info => $_ );
    }

    if ( $rows != 1 ) {
        $kernel->post( 'logger',
                error => "DB did not return one row for $handler ("
              . join( ', ', @bind_params )
              . "), but $rows. Will NOT notify!" );
        return;
    }
    elsif ( $handler eq 'closeFile' ) {

        # Notify Tier0 by creating an entry in the notify logfile
        return if $stream =~ /_DontNotifyT0$/;    #skip if DontNotify

        $kernel->post(
            'notify',
            info => join(
                ' ',
                'notifyTier0.pl',
                grep defined,
                map { exists $args->{$_} ? "--$_=$args->{$_}" : undef }
                  qw( APPNAME APPVERSION RUNNUMBER LUMISECTION FILENAME
                  PATHNAME HOSTNAME DESTINATION SETUPLABEL
                  STREAM TYPE NEVENTS FILESIZE CHECKSUM
                  HLTKEY INDEX STARTTIME STOPTIME )
            )
        );
    }
}

# Insert the line into the DB (new file)
sub insert_file {
    my ( $kernel, $args ) = @_[ KERNEL, ARG0 ];

    $args->{PRODUCER} = 'StorageManager';
    $kernel->yield(
        update_db  => $args,
        insertFile => qw(
          FILENAME PATHNAME HOSTNAME SETUPLABEL STREAM TYPE
          PRODUCER APPNAME APPVERSION RUNNUMBER LUMISECTION
          FILECOUNTER INSTANCE STARTTIME
          )
    );
    $kernel->yield(
        update_db  => $args,
        insertFileSummaryProc => qw( FILENAME )
    );
    $kernel->yield(
        update_db  => $args,
        insertFileInstancesProc => qw( FILENAME )
    );
}

# Inserts the line into the DB (closed file)
sub close_file {
    my ( $kernel, $args ) = @_[ KERNEL, ARG0 ];

    # index file name and size
    $args->{INDFILE} = $args->{FILENAME};
    $args->{INDFILE} =~ s/\.dat$/\.ind/;
    $args->{INDFILESIZE} = -1;
    if ( $host eq $args->{HOSTNAME} ) {
        if ( -e "$args->{PATHNAME}/$args->{INDFILE}" ) {
            $args->{INDFILESIZE} = ( stat(_) )[7];
        }
        elsif ( $nofilecheck == 0 ) {
            $args->{INDFILE} = '';
        }
    }

    $args->{DESTINATION} = 'Global';
    my $setuplabel = $args->{SETUPLABEL};
    my $stream     = $args->{STREAM};
    if (   $setuplabel =~ /TransferTest/
        || $stream =~ /_TransferTest$/ )
    {
        $args->{DESTINATION} = 'TransferTest';    # transfer but delete after
    }
    elsif ( $stream =~ /_NoRepack$/ || $stream eq 'Error' ) {
        $args->{DESTINATION} = 'GlobalNoRepacking';    # do not repack
        $args->{INDFILE}     = '';
        $args->{INDFILESIZE} = -1;
    }

    $kernel->yield(
        update_db => $args,
        closeFile => qw(
          FILENAME PATHNAME DESTINATION NEVENTS FILESIZE
          CHECKSUM STOPTIME INDFILE INDFILESIZE COMMENT
          )
    );
    $kernel->yield(
        update_db  => $args,
        closeFileSummaryProc => qw( FILENAME )
    );
    $kernel->yield(
        update_db  => $args,
        closeFileInstancesProc => qw( FILENAME )
    );

    # Alias index for Tier0
    $args->{INDEX} = $args->{INDFILE} if exists $args->{INDFILE};

    # Run the hook
    $kernel->yield( start_hook => $args );
}

# Queue the hook
sub start_hook {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    my $cmd = $ENV{'SM_HOOKSCRIPT'};
    return unless $cmd;
    my @args = grep defined,
      map { exists $args->{$_} ? "--$_=$args->{$_}" : undef }
      qw( APPNAME APPVERSION RUNNUMBER LUMISECTION FILENAME
      PATHNAME HOSTNAME DESTINATION SETUPLABEL
      STREAM TYPE NEVENTS FILESIZE CHECKSUM INSTANCE
      HLTKEY INDEX STARTTIME STOPTIME FILECOUNTER );
    unshift @args, $cmd;
    push @{ $heap->{task_list} }, \@args;
    $kernel->yield('next_hook');
}

# Creates a new wheel to start the hook
sub next_hook {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    while ( keys( %{ $heap->{task} } ) < $maxhooks ) {
        my $args = shift @{ $heap->{task_list} };
        last unless defined $args;
        $kernel->post( 'logger',
            debug => 'Running hook: ' .
              keys( %{ $heap->{task} } )
              . "/$maxhooks): "
              . join( ' ', @$args ) );
        my $task = POE::Wheel::Run->new(
            Program      => sub { system(@$args); },
            StdoutFilter => POE::Filter::Line->new(),
            StderrFilter => POE::Filter::Line->new(),
            StdoutEvent  => 'hook_result',
            StderrEvent  => 'hook_error',
            CloseEvent   => 'hook_done',
        );
        $heap->{task}->{ $task->ID } = $task;
        $kernel->sig_child( $task->PID, "sig_child" );
    }
}

# Catch and display information from the hook's STDOUT.
sub hook_result {
    my ( $kernel, $result ) = @_[ KERNEL, ARG0 ];
    $kernel->post( 'logger', debug => "Hook output: $result" );
}

# Catch and display information from the hook's STDERR.
sub hook_error {
    my ( $kernel, $result ) = @_[ KERNEL, ARG0 ];
    $kernel->post( 'logger', info => "Hook error: $result" );
}

# The task is done.  Delete the child wheel, and try to start a new
# task to take its place.
sub hook_done {
    my ( $kernel, $heap, $task_id ) = @_[ KERNEL, HEAP, ARG0 ];
    delete $heap->{task}->{$task_id};
    $kernel->yield("next_hook");    # See if there is something more to be done
}

# Detect the CHLD signal as each of our children exits.
sub sig_child {
    my ( $heap, $sig, $pid, $exit_val ) = @_[ HEAP, ARG0, ARG1, ARG2 ];
    my $details = delete $heap->{$pid};
}

# Got a new line in a logfile
sub got_log_line {
    my ( $kernel, $heap, $line, $wheelID ) = @_[ KERNEL, HEAP, ARG0, ARG1 ];
    my $file = $heap->{watchlist}->{$wheelID};
    $kernel->post( 'logger', debug => "In $file, got line: $line" );
    if ( $line =~ /(?:(insert|close)File)/i ) {
        $kernel->yield( parse_line => $1 => $line );
    }
    elsif ( $line =~ /^Timestamp:/ ) {
        if ( $line =~ s/\tBoR$// ) {
            $kernel->yield( parse_lumi_line => got_begin_of_run => $line );
        }
        elsif ( $line =~ s/\tEoR$// ) {
            $kernel->yield( parse_lumi_line => got_end_of_run => $line );
        }
        else {
            $kernel->yield( parse_lumi_line => got_end_of_lumi => $line );
        }
    }
    else {
        $kernel->post( 'logger', info => "Got unknown log line: $line" );
    }
}

# Splits the line by tabs and builds a hash with the result (key:value)
sub parse_lumi_line {
    my ( $kernel, $heap, $callback, $line ) = @_[ KERNEL, HEAP, ARG0, ARG1 ];
    $kernel->post( 'logger',
        debug => "Got lumi line (callback: $callback): $line" );
    return unless $callback && $line;
    my %hash = map { split /:/ } split /\t/, $line;
    if ( grep !defined, @hash{qw(Timestamp run instance host)} ) {
        $kernel->post( 'logger', warning => "Got unknown lumi line: $line" );
        return;
    }
    $hash{Timestamp} = gettimestamp( $hash{Timestamp} );
    if ( $callback =~ /_of_run$/ ) {
        $kernel->yield( get_num_sm => $callback, \%hash );
    }
    else {
        $kernel->yield( $callback, \%hash );
    }
}

# insert into RUNS values (137605, 2, 8, 0, 1, '2010-06-11 19:01:52');
sub got_begin_of_run {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    $kernel->yield(
        update_db  => $args,
        beginOfRun => qw( run instance host
          n_instances Timestamp )
    );
}

# update RUNS set (STATUS = 0, N_LUMISECTIONS = 3065, END_TIME = '2010-06-11 02:10:10')
# where RUNNUMBER = 137605 and INSTANCE = 2
sub got_end_of_run {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    $kernel->yield(
        update_db => $args,
        endOfRun  => qw( LastLumi Timestamp run instance )
    );
}

# insert into values STREAMS (137605, 767, 'Calibration',   2, 1, '2010-06-12 00:00:02');
sub got_end_of_lumi {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    for my $stream (
        sort grep { !/^(?:Timestamp|run|LS|instance|host)$/ }
        keys %$args
      )
    {
        my %localArgs = ( %$args, stream => $stream );
        $kernel->yield(
            update_db => \%localArgs,
            endOfLumi => qw( run LS stream instance Timestamp ),
            $stream
        );
    }
}

sub got_log_rollover {
    my ( $kernel, $heap, $wheelID ) = @_[ KERNEL, HEAP, ARG0 ];
    my $file = $heap->{watchlist}->{$wheelID};
    $kernel->post( 'logger', info => "$file rolled over" );
}

# Create a watcher for a file, if none exist already
sub read_changes {
    my ( $kernel, $heap, $file ) = @_[ KERNEL, HEAP, ARG0 ];

    return if $heap->{watchlist}->{$file};    # File is already monitored
    my $seek = $heap->{offsets}->{$file} || 0;
    my $size = ( stat $file )[7];
    if ( $seek > $size ) {
        $kernel->post( 'logger',
            warning =>
"Saved seek ($seek) is greater than current filesize ($size) for $file"
        );
        $seek = 0;
    }
    $kernel->post( 'logger', info => "Watching $file, starting at $seek" );
    my $wheel = POE::Wheel::FollowTail->new(
        Filename   => $file,
        InputEvent => "got_log_line",
        ResetEvent => "got_log_rollover",
        Seek       => $seek,
    );
    $heap->{watchlist}->{ $wheel->ID } = $file;
    $heap->{watchlist}->{$file} = $wheel;
}

# Some terminal signal got received
sub sig_abort {
    my ( $kernel, $heap, $signal ) = @_[ KERNEL, HEAP, ARG0 ];
    $kernel->post( 'logger', info => "Shutting down on signal SIG$signal" );
    $kernel->yield('save_offsets');
    $kernel->yield('shutdown');
    $kernel->sig_handled;
}

# Clean shutdown
sub shutdown {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    if ( $heap->{dbh} ) {
        $kernel->call( 'logger', info => "Disconnecting from the DB" );
        unless ( $heap->{dbh}->{AutoCommit} ) {
            $heap->{dbh}->commit
              or $kernel->call( 'logger', warning => "Commit failed!" );
        }
        unless ( $heap->{dbh}->disconnect ) {
            $kernel->call( 'logger',
                warning => "Disconnection from Oracle failed: $DBI::errstr" );
        }
        else {
            delete $heap->{dbh};
        }
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
        $kernel->post( 'logger', debug => "$name was modified\n" );
        $kernel->yield( read_changes => $name );
    }
    else {
        $kernel->post( 'logger', warning => "$name is no longer mounted" )
          if $event->IN_UNMOUNT;
        $kernel->post( 'logger', error => "events for $name have been lost" )
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
                if ( $key =~ /^(?:hlt)?(?:dbi|reader|phrase)$/ ) {
                    $heap->{$key} = $value;
                    next;
                }

                $kernel->post( 'logger',
                    warning =>
                      "Ignoring unknown configuration variable: $key" );
            }
        }
        close $fh;
    }
    unless ( $heap->{reader} && $heap->{phrase} ) {
        die "No DB configuration. Aborting.";
    }
    $kernel->post( 'logger', info => "Read DB configuration from $config" );
}

# Save the offset for each file so processing can be resumed at any time
sub save_offsets {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    $kernel->post( 'logger', debug => "Saving offsets" );
    $kernel->delay( save_offsets => $savedelay );
    for my $tailor ( grep { /^[0-9]+$/ } keys %{ $heap->{watchlist} } ) {
        my $file   = $heap->{watchlist}->{$tailor};
        my $wheel  = $heap->{watchlist}->{$file};
        my $offset = $wheel->tell;
        $heap->{offsets}->{$file} = $offset;
    }

    # XXX Use a ReadWrite wheel
    return unless keys %{ $heap->{offsets} };
    open my $save, '>', $offsetfile or die "Can't open $offsetfile: $!";
    for my $file ( sort keys %{ $heap->{offsets} } ) {
        my $offset = delete $heap->{offsets}->{$file};
        $kernel->post( 'logger',
            debug => "Saving offset information for $file: $offset" );
        print $save "$file $offset\n";
    }
    close $save;
}

# Read offsets, that is set the offset for each file to continue
# processing
sub read_offsets {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    return unless -s $offsetfile;
    $kernel->post( 'logger', debug => "Reading offset file $offsetfile..." );

# XXX Use a ReadWrite wheel like on
# http://github.com/bingos/poe/raw/22d59d963996d83a93fcb292c269ffbedd0d0965/docs/small-programs/reading-filehandle.pl
    open my $save, '<', $offsetfile or die "Can't open $offsetfile: $!";
    while (<$save>) {
        next unless /^(\S+) ([0-9]+)$/;
        my ( $file, $offset ) = ( $1, $2 );
        if ( -f $file ) {
            my $fsize = ( stat(_) )[7];
            if ( $offset != $fsize ) {
                $kernel->post( 'logger',
                    debug =>
                      "File $file has a different size: $offset != $fsize" );
                $kernel->yield( read_changes => $file );
            }
            $heap->{offsets}->{$file} = $offset;
            $kernel->post( 'logger',
                debug => "Setting offset information for $file: $offset" );
        }
        else {
            $kernel->post( 'logger',
                debug => "Discarding offset information"
                  . " for non-existing $file: $offset" );
        }
    }
    close $save;
}

# Print some heartbeat at fixed interval
sub heartbeat {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $message = gettimestamp( time() ) . ' Still alive in main loop.';
    $message .= ' Kernel has ' . $kernel->get_event_count()
        . ' events to process';
    $kernel->post( 'logger', info => $message );
    $kernel->delay( heartbeat => + $heartbeat );
}

# Do something with all POE events which are not caught
sub handle_default {
    my ( $kernel, $event, $args ) = @_[ KERNEL, ARG0, ARG1 ];
    print STDERR "WARNING: Session "
      . $_[SESSION]->ID
      . "caught unhandled event $event with (@$args).";
    $kernel->post( 'logger',
            warning => "Session "
          . $_[SESSION]->ID
          . " caught unhandled event $event with (@$args)." );
}
