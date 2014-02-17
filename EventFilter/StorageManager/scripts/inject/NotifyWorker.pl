#!/usr/bin/env perl
# $Id: NotifyWorker.pl,v 1.10 2012/10/07 21:56:25 babar Exp $
# --
# NotifyWoker.pl
# Monitors a directory, and sends notifications to Tier0
# according to entries in the files
# --
# Original script by Olivier Raginel <babar@cern.ch>

use strict;
use warnings;

use Linux::Inotify2;
use POE qw( Wheel::FollowTail Component::Log4perl );
use POSIX qw( strftime );
use File::Basename;
use Getopt::Long;
use YAML;
use JSON qw( to_json );
use T0::Logger::Sender;

# check arguments
unless ( $#ARGV == 2 ) {
    die "Syntax: ./NotifyWorker.pl inputpath logpath configfile";
}

my ( $inpath, $logpath, $config ) = @ARGV;
if ( -f $inpath ) {
    die "Error: this version of NotifyWorker only supports path.\n"
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
my $offsetfile = $logpath . '/notify-offset.txt';
my $heartbeat = 300;    # Print a heartbeat every 5 minutes
my $savedelay = 300;    # Frequency to save offset file, in seconds
my $log4perlConfig = '/opt/injectworker/inject/log4perl.conf';
my $t0config      = $ENV{T0_CONFIG};         # XXX Should merge both configs
my $notifySession = 'NotifyWorkerSession';   # Alias for the Client::TCP session
my $reconnectBackOff = 3;                    # Time to wait before reconnecting

# To rotate logfiles daily
sub get_logfile {
    return strftime "$logpath/$_[0]-%Y%m%d-$host.log", localtime time;
}

# Create logger
Log::Log4perl->init_and_watch( $log4perlConfig, 'HUP' );
POE::Component::Log4perl->spawn(
    Alias      => 'logger',
    Category   => 'NotifyWorker',
    ConfigFile => $log4perlConfig,
    GetLogfile => \&get_logfile,
);

# Create TCP client to send notifications to the Copy Manager
$notifySession = T0::Logger::Sender->new(
    Name    => $notifySession,
    Config  => $t0config,
    OnError => \&notify_OnError,
)->{Name};

# Start POE Session, which will do everything
POE::Session->create(
    inline_states => {
        _start       => \&start,
        inotify_poll => sub {
            $_[HEAP]{inotify}->poll;
        },
        watch_hdlr       => \&watch_hdlr,
        save_offsets     => \&save_offsets,
        parse_line       => \&parse_line,
        read_offsets     => \&read_offsets,
        read_changes     => \&read_changes,
        got_log_line     => \&got_log_line,
        got_log_rollover => \&got_log_rollover,
        notify_tier0     => \&notify_tier0,
        shutdown         => \&shutdown,
        heartbeat        => \&heartbeat,
        switch_file      => \&switch_file,
        set_rotate_alarm => \&set_rotate_alarm,
        setup_lock       => \&setup_lock,
        notify_success   => \&notify_success,
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
    return strftime "%Y-%m-%d %H:%M:%S", localtime time;
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

    $kernel->post( 'logger', info => "Entering main while loop now" );
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
            $kernel->post( 'logger',
                warn => "Warning: Lock \"$lockfile\""
                  . "exists, pid $pid (NOT running). Removing stale lock file?"
            );
        }
    }
    open my $fh, '>', $lockfile or die "Cannot create $lockfile: $!";
    print $fh "$$\n";    # Fill it with pid
    close $fh;
    $heap->{LockFile} = $lockfile;
    $kernel->post( 'logger', info => "Set lock to $lockfile for $$" );

    $kernel->sig( INT  => 'sig_abort' );
    $kernel->sig( TERM => 'sig_abort' );
    $kernel->sig( QUIT => 'sig_abort' );
}

# This makes the bridge between the global default session, which parses the
# logfiles, and the Client::TCP which communicates with the CopyManager
sub notify_tier0 {
    my ( $kernel, $session, $heap, $args, $wheelID, $offset ) =
      @_[ KERNEL, SESSION, HEAP, ARG0 .. ARG2 ];
    $args->{DAQFileClosed}    = 1;
    $args->{T0FirstKnownTime} = time;
    my $nodbupdate = delete $args->{nodbupdate};
    $kernel->post( 'logger', info => "Notifying Tier0: " . to_json($args) );
    $kernel->call( $notifySession, send => $args );
    return if $nodbupdate;
    $kernel->call(
        $notifySession,
        send => {
            TransferStatus => '1',
            STATUS         => 'new',
            FILENAME       => $args->{FILENAME},
        }
    );

    # Should check something...
    $kernel->yield( 'notify_success', $wheelID, $offset );
}

# Badly named, as we have no idea of the outcome (yet) XXX
sub notify_success {
    my ( $kernel, $heap, $wheelID, $offset ) = @_[ KERNEL, HEAP, ARG0 .. ARG1 ];

    # Update offset, file has been processed
    if ( defined $wheelID && defined $offset ) {
        my $current = $heap->{offset}->{$wheelID};
        if ( $current > $offset ) {
            my $file = $heap->{watchlist}->{$wheelID};
            $kernel->post( 'logger',
                warning =>
                  "$file was processed backwards: $offset < $current!" );
        }
        else {
            $heap->{offset}->{$wheelID} =
              $offset;    # File processed up to this offset
        }
    }
    else {
        $kernel->post( 'logger', warning => "No offset information" );
    }
}

# This fonction is run on the Client::TCP session
sub notify_OnError {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    $kernel->post( 'logger',
        warning =>
          "ERROR: could not connect to remote end, file not injected" );
    $kernel->delay( reconnect => $reconnectBackOff );
}

# Parse lines like
# [2012/10/04 18:20:04] INFO notifyTier0.pl --APPNAME=CMSSW --APPVERSION=CMSSW_3_2_1_onlpatch4_ONLINE --RUNNUMBER=64236 --LUMISECTION=1 --FILENAME=TransferTestWithSafety.00064236.1.A.storageManager.21.5.dat --PATHNAME=/store/babar/global/closed/ --HOSTNAME=dvsrv-C2F37-01 --DESTINATION=TransferTest --SETUPLABEL=TransferTestWithSafety --STREAM=A --TYPE=streamer --NEVENTS=4 --FILESIZE=503071 --CHECKSUM=c93da794 --HLTKEY=DumbDBTest --STARTTIME=1349367604 --STOPTIME=1349367604
sub parse_line {
    my ( $kernel, $heap, $callback, $line, $wheelID, $offset ) =
      @_[ KERNEL, HEAP, ARG0 .. ARG3 ];
    return
      unless $line =~ s/^\[[^\]]*] INFO (?:closeFile|notifyTier0)\.pl //
      ;    # Remove [date] INFO notifyTier0.pl
    my %args = map { /^--(.*)=(.*)$/ ? ( $1, $2 ) : $_ } split / +/, $line;

    # Maintain compatibility
    $args{START_TIME}  = delete $args{STARTTIME};
    $args{STOP_TIME}   = delete $args{STOPTIME};
    $args{APP_VERSION} = delete $args{APPVERSION};
    $args{APP_NAME}    = delete $args{APPNAME};

    $kernel->yield( $callback => \%args, $wheelID => $offset );
}

sub got_log_line {
    my ( $kernel, $heap, $line, $wheelID ) = @_[ KERNEL, HEAP, ARG0, ARG1 ];
    my $file   = $heap->{watchlist}->{$wheelID};
    my $offset = $heap->{watchlist}->{$file}->tell();
    $kernel->post( 'logger', debug => "In $file, got line: $line" );
    if ( $line =~ /(?:closeFile|notifyTier0)/i ) {
        $kernel->yield(
            parse_line => notify_tier0 => $line,
            $wheelID   => $offset
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

    # XXX Would be great not to use a FollowTail wheel, but to use inotify
    return if $heap->{watchlist}->{$file};    # File is already monitored
    my $seek = $heap->{offset}->{$file} || 0;
    my $size = ( stat $file )[7];
    if ( $seek > $size ) {
        $kernel->post( 'logger',
            warning => "Saved seek ($seek) is greater than"
              . " current filesize ($size) for $file" );
        $seek = 0;
    }
    $kernel->post( 'logger', info => "Watching $file, starting at $seek" );
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
    $kernel->post( 'logger', info => "Shutting down on signal SIG$signal" );
    $kernel->yield('save_offsets');
    $kernel->yield('shutdown');
    $kernel->sig_handled;
}

# Clean shutdown
sub shutdown {
    my ( $kernel, $heap ) = @_[ KERNEL, HEAP ];
    my $lockfile = $heap->{LockFile};
    unlink $lockfile if $lockfile;
    die "Shutting down!";
}

# postback called when some iNotify event is raised
sub watch_hdlr {
    my $kernel = $_[KERNEL];
    my $event  = $_[ARG1][0];
    my $name   = $event->fullname;

    # Filter only notify-YYYYMMDD-srv-C2C0x-yy.log files
    return unless $event->name =~ /^notify-\d{8}-$host\.log$/;
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
    $kernel->post( 'logger', debug => "Reading offset file $offsetfile..." );

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
            $heap->{offset}->{$file} = $offset;
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
    $message .=
      ' Kernel has ' . $kernel->get_event_count() . ' events to process';
    $kernel->post( 'logger', info => $message );
    $kernel->delay( heartbeat => $heartbeat );
}

# Do something with all POE events which are not caught
sub handle_default {
    my ( $kernel, $event, $args ) = @_[ KERNEL, ARG0, ARG1 ];
    print STDERR "WARNING: Session "
      . $_[SESSION]->ID
      . " caught unhandled event $event with (@$args).";
    $kernel->post( 'logger',
            warning => "Session "
          . $_[SESSION]->ID
          . " caught unhandled event $event with args (@$args)." );
}
