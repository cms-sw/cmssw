#!/usr/bin/env perl
# $Id: InjectWorker.pl,v 1.89 2012/10/07 21:56:25 babar Exp $
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
use FindBin;
use DBI;
use Getopt::Long;

################################################################################
my $nodbint     = 0; # SM_DONTACCESSDB: no access any DB at all
my $nodbwrite   = 0; # SM_DONTWRITEDB : no write to DB but retrieve HLT key
my $norunconddb = 0; # SM_NORUNCONDDB : do not access ConfDB at all, fake
my $nofilecheck = 0; # SM_NOFILECHECK : no check if files are locally accessible
my $maxhooks    = 1; # Number of parallel hooks
################################################################################
my %invalidOracleError = (
    '03113' => 'End-of-file on communication channel',
    '03114' => 'not connected to Oracle',
    '03135' => 'Connection lost contact',
    '01031' => 'insufficient privileges',
    '01012' => 'not logged on',
    '01003' => 'no statement parsed',
    '02291' => 'integrity constraint violated - parent key not found',
    '12545' => 'target host or object does not exist',
    '17008' => 'closed connection',
    '25401' => 'can not continue fetches',
    '25402' => 'transaction must roll back',
    '25403' => 'could not reconnect',
    '25404' => 'lost instance',
    '25405' => 'transaction status unknown',
    '25406' => 'could not generate a connect address',
    '25407' => 'connection terminated',
    '25408' => 'can not safely replay call',
    '25409' =>
      'failover happened during the network operation, cannot continue',
);

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

if ( defined $ENV{'SM_NORUNCONDDB'} ) {
    $norunconddb = 1;
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

##############################################################################
# Configuration
chomp( my $host = `hostname -s` );
my $heartbeat  = 300;    # Print a heartbeat every 5 minutes
my $savedelay  = 300;    # Frequency to save offset file, in seconds
my $retrydelay = 30;     # Backoff time before retrying a DB query, in seconds
my $maxretries = 10;     # Maximum number of DB retries
my $dbbackoff  = 10;     # Seconds to wait between 2 DB connection tries
my $offsetfile     = $logpath . '/offset.txt';
my $log4perlConfig = $FindBin::Bin . '/log4perl.conf';

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
        watch_hdlr            => \&watch_hdlr,
        save_offsets          => \&save_offsets,
        update_db             => \&update_db,
        read_db_config        => \&read_db_config,
        setup_db              => \&setup_db,
        setup_main_db         => \&setup_main_db,
        setup_runcond_db      => \&setup_runcond_db,
        get_num_sm            => \&get_num_sm,
        get_hlt_key           => \&get_hlt_key,
        get_from_runcond      => \&get_from_runcond,
        start_hook            => \&start_hook,
        next_hook             => \&next_hook,
        hook_result           => \&hook_result,
        hook_error            => \&hook_error,
        hook_done             => \&hook_done,
        parse_line            => \&parse_line,
        parse_lumi_line       => \&parse_lumi_line,
        got_begin_of_run      => \&got_begin_of_run,
        got_end_of_run        => \&got_end_of_run,
        check_eor_consistency => \&check_eor_consistency,
        got_end_of_lumi       => \&got_end_of_lumi,
        read_offsets          => \&read_offsets,
        read_changes          => \&read_changes,
        got_log_line          => \&got_log_line,
        got_log_rollover      => \&got_log_rollover,
        insert_file           => \&insert_file,
        close_file            => \&close_file,
        shutdown              => \&shutdown,
        heartbeat             => \&heartbeat,
        switch_file           => \&switch_file,
        set_rotate_alarm      => \&set_rotate_alarm,
        setup_lock            => \&setup_lock,
        sig_child             => \&sig_child,
        sig_abort             => \&sig_abort,
        _stop                 => \&shutdown,
        _default              => \&handle_default,
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

# Switches logfiles daily
sub switch_file {
    my $kernel = $_[KERNEL];
    $kernel->yield('set_rotate_alarm');
    my %appenderPrefix = (
        InjectLogfile       => 'log',
        NotificationLogfile => 'notify',
        NotifyLogfile       => 'lognotify'
    );
    my $appList = Log::Log4perl->appenders();
    for my $appender ( keys %$appList ) {
        my $app = $appList->{$appender};
        $app->file_switch( get_logfile( $appenderPrefix{$appender} ) );
    }
}

# Pseudo-hack to rotate files daily
sub set_rotate_alarm {
    my $kernel = $_[KERNEL];
    my ( $sec, $min, $hour ) = localtime;
    my $wakeme = time + 86400 + 1 - ( $sec + 60 * ( $min + 60 * $hour ) );
    $kernel->call( 'logger',
        info =>
          strftime( "Set alarm for %Y-%m-%d %H:%M:%S", localtime $wakeme ) );
    $kernel->alarm( switch_file => $wakeme );
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

    # Rotate logfiles daily
    $kernel->yield('set_rotate_alarm');

    $kernel->call( 'logger', info => "Entering main while loop now" );
}

# lockfile
sub setup_lock {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $lockfile = '.' . basename( $0, '.pl' ) . ".lock";
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
              "Error: Lock \"$lockfile\" exists, pid $pid (running: $process)."
              . " Stale lock file?";
        }
        else {
            $kernel->call( 'logger',
                warn => "Warning: Lock \"$lockfile\""
                  . "exists, pid $pid (NOT running). Removing stale lock file?"
            );
        }
    }
    open my $fh, '>', $lockfile or die "Cannot create $lockfile: $!";
    print $fh "$$\n";    # Fill it with pid
    close $fh;
    $heap->{LockFile} = $lockfile;
    $kernel->call( 'logger', info => "Set lock to $lockfile for $$" );

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
        $kernel->call( 'logger', warning => "No DB (even!) access flag set" );
    }
    else {
        $kernel->yield( 'read_db_config', $config );
        $kernel->yield('setup_runcond_db');
        if ($nodbwrite) {

            # XXX This will not work as there is no further code for that case
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
        debug => "Setting up DB connection for $heap->{dbi}"
          . " and $heap->{reader} write access" );
    my $failed = 0;
    my $last_failed = $heap->{last_failed_main_db} || 0;
    return if time - $last_failed < $dbbackoff;
    $heap->{dbh} = DBI->connect_cached( @$heap{qw( dbi reader phrase )} )
      or $failed++;
    if ($failed) {
        $kernel->call( 'logger',
            error => "Connection to $heap->{dbi} failed: $DBI::errstr" );
        $kernel->delay( setup_main_db => $dbbackoff );
        $heap->{last_failed_main_db} = time;
        return;
    }

    # Enable DBMS to get statistics
    $heap->{dbh}->func( 1000000, 'dbms_output_enable' );

    # For new files
    my $sql =
        "INSERT INTO CMS_STOMGR.FILES_CREATED "
      . "(FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION,"
      . "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME) "
      . "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $heap->{sths}->{insertFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For new files: update SM_SUMMARY
    $sql = "BEGIN CMS_STOMGR.FILES_CREATED_PROC_SUMMARY( ? ); END;";
    $heap->{sths}->{insertFileSummaryProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For new files: update SM_INSTANCES
    $sql = "BEGIN CMS_STOMGR.FILES_CREATED_PROC_INSTANCES( ? ); END;";
    $heap->{sths}->{insertFileInstancesProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files
    $sql =
        "INSERT INTO CMS_STOMGR.FILES_INJECTED "
      . "(FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,INDFILENAME,INDFILESIZE,COMMENT_STR) "
      . "VALUES (?,?,?,?,?,?,"
      . "TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'),?,?,?)";
    $heap->{sths}->{closeFile} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files: update SM_SUMMARY
    $sql = "BEGIN CMS_STOMGR.FILES_INJECTED_PROC_SUMMARY( ? ); END;";
    $heap->{sths}->{closeFileSummaryProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    # For closed files: update SM_INSTANCES
    $sql = "BEGIN CMS_STOMGR.FILES_INJECTED_PROC_INSTANCES( ? ); END;";
    $heap->{sths}->{closeFileInstancesProc} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

  # For explanation, see:
  # https://twiki.cern.ch/twiki/bin/viewauth/CMS/StorageManagerEndOfLumiHandling
    $sql =
        "insert into CMS_STOMGR.RUNS "
      . "(RUNNUMBER, INSTANCE, HOSTNAME, N_INSTANCES, N_LUMISECTIONS, "
      . " STATUS, MAX_LUMISECTION, LAST_CONSECUTIVE, START_TIME)"
      . " values "
      . "(        ?,        ?,        ?,           ?,              0, "
      . "      1,            NULL,             NULL, TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))";
    $heap->{sths}->{beginOfRun} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "update CMS_STOMGR.RUNS set "
      . "STATUS = 0, N_LUMISECTIONS = ?, "
      . "END_TIME = TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'), "
      . "MAX_LUMISECTION = ?, LAST_CONSECUTIVE = ? "
      . "where RUNNUMBER = ? and INSTANCE = ?";
    $heap->{sths}->{endOfRun} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "insert into CMS_STOMGR.STREAMS "
      . "(RUNNUMBER, LUMISECTION, STREAM, INSTANCE,"
      . " CTIME,                           EOLS, FILECOUNT)"
      . " values "
      . "(        ?,           ?,      ?,        ?,"
      . " TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'), ?,         ?)";
    $heap->{sths}->{endOfLumi} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

    $sql =
        "insert into CMS_STOMGR.BAD_RUNS "
      . "(RUNNUMBER, INSTANCE, LUMISECTION)"
      . " values "
      . "(        ?,        ?,           ?)";
    $heap->{sths}->{badLumi} = $heap->{dbh}->prepare($sql)
      or die "Error: Prepare failed for $sql: " . $heap->{dbh}->errstr;

}

# Setup the run conditions DB connection
sub setup_runcond_db {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];

    return if $norunconddb;    # Skip for daqval

    # this is for HLT key and number of SM queries
    $kernel->call( 'logger',
        debug => "Setting up DB connection for $heap->{dbi}"
          . " and $heap->{reader} read access" );

    my $failed = 0;
    unless ( $heap->{dbh} ) {
        $heap->{dbh} = DBI->connect_cached( @$heap{qw( dbi reader phrase )} )
          or $failed++;
    }
    if ($failed) {
        $kernel->call( 'logger',
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
        $callback,
        'CMS.DAQ:NB_ACTIVE_STORAGEMANAGERS_T', $args
    );
}

# Retrieve the HLT key for a given run
sub get_hlt_key {
    my ( $kernel, $heap, $kind, $args ) = @_[ KERNEL, HEAP, ARG0 .. ARG2 ];
    $kernel->yield(
        get_from_runcond => 'HLTkey',
        "${kind}_file",
        'CMS.LVL0:HLT_KEY_DESCRIPTION', $args
    );
}

sub get_from_runcond {
    my ( $kernel, $heap, $kind, $callback, $key, $args ) =
      @_[ KERNEL, HEAP, ARG0 .. ARG3 ];

    my $runnumber = $args->{RUNNUMBER} || $args->{run};
    unless ( defined $runnumber and $kind and $key ) {
        $kind      = 'Undefined kind of query' unless $kind;
        $key       = 'Undefined key'           unless $key;
        $runnumber = 'Undefined runnumber'     unless defined $runnumber;
        $kernel->call( 'logger',
            error =>
              "Trying to get $kind (key: $key) for runnumber $runnumber!" );
        return;
    }
    my $cached = $heap->{$kind}->{$runnumber};    # Try to get it from the cache

    # Fake for daqval
    if ($norunconddb) {
        $cached = $heap->{$kind}->{$runnumber} =
          $kind eq 'HLTkey' ? 'DumbDBTest' : 2;
    }

    # query run conditions db if key was not already obtained for this run
    my $sth = $heap->{sths}->{runCond};
    if ( !defined $cached && $sth ) {
        my $errflag = 0;
        $kernel->call( 'logger',
            debug => "Querying DB for runnumber $runnumber" );
        $sth->execute( $runnumber, $key ) or $errflag = 1;
        if ($errflag) {
            $kernel->call( 'logger',
                error =>
                  "DB query get $kind (key: $key) for run $runnumber returned "
                  . $sth->errstr );
        }
        else {
            ( $heap->{$kind}->{$runnumber} ) = ($cached) = $sth->fetchrow_array
              or $errflag = 1;
            if ($errflag) {
                $kernel->call( 'logger',
                    error =>
                      "Fetching $kind (key: $key) for run $runnumber returned "
                      . $sth->errstr );
            }
            else {
                $kernel->call( 'logger',
                    debug =>
                      "Obtained $kind key $key = $cached for run $runnumber" );
            }
        }
        $sth->finish;
    }
    unless ( defined $cached ) {
        my $delay = $args->{_RetryDelay} ||= $retrydelay;
        my $retries =
          ( $args->{_Retries} =
              defined $args->{_Retries} ? $args->{_Retries} : $maxretries )--;
        if ($retries) {
            $kernel->call( 'logger',
                error =>
                  "Could not retrieve $kind (key: $key) for run $runnumber."
                  . " Retrying ($retries left) in $delay" );
            $kernel->delay_add(
                get_from_runcond => $delay,
                $kind, $callback, $key, $args
            );
        }
        else {
            $kernel->call( 'logger',
                error =>
                  "Could not retrieve $kind (key: $key) for run $runnumber."
                  . " Giving up after $maxretries tries." );
        }
        return;
    }
    if ( $kind eq 'HLTkey' ) {
        $args->{COMMENT} = 'HLTKEY=' . $cached;
        $args->{HLTKEY}  = $cached;
    }
    else {
        $args->{n_instances} = $cached;
    }
    $kernel->yield( $callback => $args );
}

# Parse lines like
#./closeFile.pl  --FILENAME Data.00133697.0135.Express.storageManager.00.0000.dat --FILECOUNTER 0 --NEVENTS 21 --FILESIZE 1508412 --STARTTIME 1271857503 --STOPTIME 1271857518 --STATUS closed --RUNNUMBER 133697 --LUMISECTION 135 --PATHNAME /store/global//02/closed/ --HOSTNAME srv-C2C06-12 --SETUPLABEL Data --STREAM Express --INSTANCE 0 --SAFETY 0 --APPVERSION CMSSW_3_5_4_onlpatch3_ONLINE --APPNAME CMSSW --TYPE streamer --DEBUGCLOSE 1 --CHECKSUM c8b5a624 --CHECKSUMIND 8912d364
sub parse_line {
    my ( $kernel, $heap, $callback, $line, $wheelID, $offset ) =
      @_[ KERNEL, HEAP, ARG0 .. ARG3 ];
    my @args = split / +/, $line;
    my $kind = shift @args;

    # Test input parameters and return if something fishy is found
    return unless $kind =~ /^\.\/${callback}File\.pl$/;
    if ( @args % 2 ) {
        $kernel->call( 'logger', error => "Could not parse line $line!" );
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
    $args{_WheelOffset} = [ $wheelID => $offset ];    # Save offset information
    $kernel->yield( get_hlt_key => $callback, \%args );
}

# injection subroutine
sub update_db {
    my ( $kernel, $session, $heap, $args, $handler, @params ) =
      @_[ KERNEL, SESSION, HEAP, ARG0 .. $#_ ];

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
            $kernel->call( 'logger',
                error => "$handler failed: Could not obtain parameter $_" );
            return;
        }
    }
    $kernel->call( 'logger',
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
      );                               #skip if NoTransfer option is set

    my $errflag = 0;
    my $rows    = $heap->{sths}->{$handler}->execute(@bind_params)
      or $errflag = 1;

    $rows = 'undef' unless defined $rows;
    if ($errflag) {
        my $errorString  = $heap->{sths}->{$handler}->errstr;
        my $errorMessage = '';
        my ($oracleError) = $errorString =~ /ORA-(\d+):/;
        if ( exists $invalidOracleError{$oracleError} ) {
            my $delay = $args->{_RetryDelay} ||= $retrydelay;
            my $retries = ( $args->{_Retries} ||= $maxretries )--;
            if ($retries) {
                $errorMessage = $invalidOracleError{$oracleError}
                  . ". Reconnecting DB & retrying ($retries left) in $delay";
                $kernel->call( $session => 'setup_main_db' );
                $kernel->delay_add(
                    update_db => $delay,
                    $args, $handler, @params,
                );
            }
            else {
                $errorMessage = " Giving up after $maxretries tries.";
            }
        }
        $kernel->call( 'logger',
                error => "DB access error (rows: $rows)"
              . " for $handler when executing ("
              . join( ', ', @bind_params )
              . '), DB returned: '
              . $errorString . '. '
              . $errorMessage );
        return;
    }

    # Print any messages from the DB's dbms output
    for ( $heap->{dbh}->func('dbms_output_get') ) {
        $kernel->call( 'logger', info => $_ );
    }

    if ( $rows != 1 ) {
        $kernel->call( 'logger',
                error => "DB did not return one row for $handler ("
              . join( ', ', @bind_params )
              . "), but $rows. Will NOT notify!" );
        return;
    }

    # If injection was successful, update the summary tables
    if ( $handler =~ /^(?:insert|close)File$/ ) {
        $kernel->call( 'logger',
            info => "$handler successfull for $args->{FILENAME}" );
        $kernel->yield(
            update_db               => $args,
            "${handler}SummaryProc" => qw( FILENAME )
        );
        $kernel->yield(
            update_db                 => $args,
            "${handler}InstancesProc" => qw( FILENAME )
        );
    }

    # Notify Tier0 by creating an entry in the notify logfile
    if ( $handler eq 'closeFile' ) {
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
                  HLTKEY STARTTIME STOPTIME )
            )
        );
    }
    if ( $args->{_WheelOffset} ) {
        my ( $wheelID, $offset ) = @{ $args->{_WheelOffset} };
        my $current = $heap->{offset}->{$wheelID};
        if ( $current > $offset ) {
            my $file = $heap->{watchlist}->{$wheelID};
            $kernel->call( 'logger',
                warning =>
                  "$file was processed backwards: $offset < $current!" );
        }
        else {
            $heap->{offset}->{$wheelID} =
              $offset;    # File processed up to this offset
        }
    }
    else {
        $kernel->call( 'logger', warning => "No offset information" );
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
      HLTKEY STARTTIME STOPTIME FILECOUNTER );
    unshift @args, $cmd;
    push @{ $heap->{task_list} }, \@args;
    $kernel->yield('next_hook');
}

# Creates a new wheel to start the hook
sub next_hook {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    while ( keys( %{ $heap->{task} } ) < $maxhooks ) {
        my $args = shift @{ $heap->{task_list} };
        last unless defined $args;
        $kernel->call( 'logger',
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
    $kernel->call( 'logger', debug => "Hook output: $result" );
}

# Catch and display information from the hook's STDERR.
sub hook_error {
    my ( $kernel, $result ) = @_[ KERNEL, ARG0 ];
    $kernel->call( 'logger', info => "Hook error: $result" );
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
    my $file   = $heap->{watchlist}->{$wheelID};
    my $offset = $heap->{watchlist}->{$file}->tell();
    $kernel->call( 'logger', debug => "In $file, got line: $line" );
    if ( $line =~ /(?:(insert|close)File)/i ) {
        $kernel->yield( parse_line => $1 => $line, $wheelID => $offset );
    }
    elsif ( $line =~ /^Timestamp:/ ) {
        if ( $line =~ s/\tBoR$// ) {
            $kernel->yield(
                parse_lumi_line => got_begin_of_run => $line,
                $wheelID        => $offset
            );
        }
        elsif ( $line =~ s/\tEoR$// ) {
            $kernel->yield(
                parse_lumi_line => got_end_of_run => $line,
                $wheelID        => $offset
            );
        }
        else {
            $kernel->yield(
                parse_lumi_line => got_end_of_lumi => $line,
                $wheelID        => $offset
            );
        }
    }
    else {
        $kernel->call( 'logger', info => "Got unknown log line: $line" );
    }
}

# Splits the line by tabs and builds a hash with the result (key:value)
sub parse_lumi_line {
    my ( $kernel, $heap, $callback, $line, $wheelID, $offset ) =
      @_[ KERNEL, HEAP, ARG0 .. ARG3 ];
    $kernel->call( 'logger',
        debug => "Got lumi line (callback: $callback): $line" );
    return unless $callback && $line;
    my %hash = map { split /:/ } split /\t/, $line;
    if ( grep !defined, @hash{qw(Timestamp run instance host)} ) {
        $kernel->call( 'logger', warning => "Got unknown lumi line: $line" );
        return;
    }
    $hash{Timestamp} =
      gettimestamp( $hash{Timestamp} );    # Change unix timestamp to string
    $hash{_WheelOffset} = [ $wheelID => $offset ];    # Save offset information
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

    # Check consistency of EOLS in case we didn't get EoR for previous run
    my $run = $args->{run};
    for my $runnumber ( keys %{ $heap->{_SeenEoLS} } ) {
        next if $runnumber == $run;    # race condition
        my $seen = delete $heap->{_SeenEoLS}->{$runnumber};
        next unless $seen->{max};      # Ignore empty run
        $kernel->call( 'logger',
            warning => "No EoR marker for run $runnumber"
              . " ($seen->{max} lumisections)" );
    }
    $heap->{_SeenEoLS}->{$run}->{0}++;    # Set fake LS 0 as processed
    $heap->{_SeenEoLS}->{$run}->{max} ||= 0;    # Set default max to 0
}

# update RUNS set (STATUS = 0, N_LUMISECTIONS = 3065, END_TIME = '2010-06-11 02:10:10')
# where RUNNUMBER = 137605 and INSTANCE = 2
sub got_end_of_run {
    my ( $kernel, $session, $heap, $args ) = @_[ KERNEL, SESSION, HEAP, ARG0 ];
    $kernel->call( $session => check_eor_consistency => $args );
    $kernel->yield(
        update_db => $args,
        endOfRun  => qw( LastLumi Timestamp maxLumi lastGood run instance )
    );
}

# Check consistency of EOLS received during the run
sub check_eor_consistency {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    my $runnumber = $args->{run};

    # Set defaults, in case we do not have enough information
    $args->{maxLumi}  = 0;
    $args->{lastGood} = 0;

    if ( my $seen = delete $heap->{_SeenEoLS}->{$runnumber} ) {
        unless ( delete $seen->{0} ) {    # In case we get EoR without a BoR
            $kernel->call( 'logger',
                warning => "No BoR marker for run $runnumber, or not seen" );

            # XXX should do something, call SP?
            return;
        }
        $args->{maxLumi} = $seen->{max};
        return unless $seen->{max};       # In case we get EoR without any EoLS
        my ( $seenBad, @badLS ) = (0);
        for my $lumisection ( 1 .. $seen->{max} ) {
            if ( delete $seen->{$lumisection} ) {
                $args->{lastGood} = $lumisection if !$seenBad;
            }
            else {
                $seenBad++;
                push @badLS, $lumisection;
            }
        }

        # Found some bad LS => run is bad
        # Log and mark it bad in the database
        if ($seenBad) {
            $kernel->call( 'logger',
                warning => "No EoLS marker for $seenBad LS of run $runnumber: "
                  . join( ', ', @badLS ) );
            for my $badLumi (@badLS) {
                my %localArgs = (
                    map { $_ => $args->{$_} } qw( run instance ),
                    badLumi => $badLumi
                );
                $kernel->post(
                    update_db => \%localArgs,
                    badLumi   => qw( run instance badLumi )
                );
            }
        }
    }
    else {
        $kernel->call( 'logger',
            warning => "No EoLS marker for run $runnumber" );

        # XXX should do something, call SP?
        return;
    }
}

# insert into values STREAMS (137605, 767, 'Calibration',   2, 1, '2010-06-12 00:00:02');
sub got_end_of_lumi {
    my ( $kernel, $heap, $args ) = @_[ KERNEL, HEAP, ARG0 ];
    $args->{EoLS} ||= 0;    # Ensure default value
    for my $stream (
        sort grep { !/^(?:Timestamp|run|LS|instance|host|EoLS|_.*)$/ }
        keys %$args
      )
    {
        my %localArgs = ( %$args, stream => $stream );
        $kernel->yield(
            update_db => \%localArgs,
            endOfLumi => qw( run LS stream instance Timestamp EoLS ),
            $stream
        );
    }

    # Mark the LS as processed, and update the max LS seen for integrity check
    my ( $runnumber, $lumisection ) = @$args{qw(run LS)};
    $heap->{_SeenEoLS}->{$runnumber}->{$lumisection}++;   # Mark it as processed
    $heap->{_SeenEoLS}->{$runnumber}->{max} =             # Update max LS
      ( $heap->{_SeenEoLS}->{$runnumber}->{max} || 0 ) > $lumisection
      ? $heap->{_SeenEoLS}->{$runnumber}->{max}
      : $lumisection;
}

sub got_log_rollover {
    my ( $kernel, $heap, $wheelID ) = @_[ KERNEL, HEAP, ARG0 ];
    my $file = $heap->{watchlist}->{$wheelID};
    $kernel->call( 'logger', info => "$file rolled over" );
}

# Create a watcher for a file, if none exist already
sub read_changes {
    my ( $kernel, $heap, $file ) = @_[ KERNEL, HEAP, ARG0 ];

    # XXX Would be great not to use a FollowTail wheel, but to use inotify
    return if $heap->{watchlist}->{$file};    # File is already monitored
    my $seek = $heap->{offset}->{$file} || 0;
    my $size = ( stat $file )[7];
    if ( $seek > $size ) {
        $kernel->call( 'logger',
            warning => "Saved seek ($seek) is greater than"
              . " current filesize ($size) for $file" );
        $seek = 0;
    }
    $kernel->call( 'logger', info => "Watching $file, starting at $seek" );
    my $wheel = POE::Wheel::FollowTail->new(
        Filename   => $file,
        InputEvent => "got_log_line",
        ResetEvent => "got_log_rollover",
        Seek       => $seek,
    );
    $heap->{offset}->{ $wheel->ID }    = $seek;     # Save offset per wheel ID
    $heap->{watchlist}->{ $wheel->ID } = $file;     # Map wheel ID => file
    $heap->{watchlist}->{$file}        = $wheel;    # Map file => wheel object
}

# Some terminal signal got received
sub sig_abort {
    my ( $kernel, $heap, $signal ) = @_[ KERNEL, HEAP, ARG0 ];
    $kernel->call( 'logger', info => "Shutting down on signal SIG$signal" );
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
    my $lockfile = $heap->{LockFile};
    unlink $lockfile if $lockfile;
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
                if ( $key =~ /^(?:hlt)?(?:dbi|reader|phrase)$/ ) {
                    $heap->{$key} = $value;
                    next;
                }

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
    my %offset;

    # Call, otherwise log is lost during shutdown
    $kernel->call( 'logger', info => "Saving offsets" );
    $kernel->delay( save_offsets => $savedelay );

    # First ensure all tailors have offset sets
    for my $tailor ( grep { /^[0-9]+$/ } keys %{ $heap->{watchlist} } ) {
        my $file  = $heap->{watchlist}->{$tailor};
        my $wheel = $heap->{watchlist}->{$file};
        $offset{$file} = $heap->{offset}->{$tailor} || $wheel->tell;
    }

    return unless keys %offset;    # Nothing to do

    # Loop over offsets, saving them to disk
    open my $save, '>', $offsetfile or die "Can't open $offsetfile: $!";
    for my $file ( sort keys %offset ) {
        my $offset = $offset{$file};
        $kernel->call( 'logger',
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
    $kernel->call( 'logger', debug => "Reading offset file $offsetfile..." );

    open my $save, '<', $offsetfile or die "Can't open $offsetfile: $!";
    while (<$save>) {
        next unless /^(\S+) ([0-9]+)$/;
        my ( $file, $offset ) = ( $1, $2 );
        if ( -f $file ) {
            my $fsize = ( stat(_) )[7];
            if ( $offset != $fsize ) {
                $kernel->call( 'logger',
                    debug =>
                      "File $file has a different size: $offset != $fsize" );
                $kernel->yield( read_changes => $file );
            }
            $heap->{offset}->{$file} = $offset;
            $kernel->call( 'logger',
                debug => "Setting offset information for $file: $offset" );
        }
        else {
            $kernel->call( 'logger',
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
    $message .=
      ' Kernel has ' . $kernel->get_event_count() . ' events to process';
    $kernel->call( 'logger', info => $message );
    $kernel->delay( heartbeat => $heartbeat );
}

# Do something with all POE events which are not caught
sub handle_default {
    my ( $kernel, $event, $args ) = @_[ KERNEL, ARG0, ARG1 ];
    print STDERR "WARNING: Session "
      . $_[SESSION]->ID
      . "caught unhandled event $event with (@$args).";
    $kernel->call( 'logger',
            warning => "Session "
          . $_[SESSION]->ID
          . " caught unhandled event $event with (@$args)." );
}
