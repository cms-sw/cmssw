#!/usr/bin/env perl
# $Id: injectFileIntoTransferSystem.pl,v 1.48 2012/02/29 15:24:23 babar Exp $

use strict;
use DBI;
use Getopt::Long;
use File::Basename;
use Sys::Hostname;

sub usageShort {
    print "
  ############################################################################################
  For more information please see help:
  $0 --help
  ############################################################################################
  \n";
    exit;
}

sub usage {
    print " 
  ############################################################################################
  Usage:
  $0 --help  : show this message

  --------------------------------------------------------------------------------------------
  For more help, https://twiki.cern.ch/twiki/bin/view/CMS/ManuallyInjectIntoTransferSystem
  --------------------------------------------------------------------------------------------

  This script will inform the Tier-0 transfer system to transfer a given file from a host
  in Cessy to Castor (DBS). In order to allow for safe copying it will insert relevant
  meta information into an online database. This database ensures the integrity of all 
  transmitted files and therefore does not allow you to transmit the same file twice.
  You therefore must be sure about all options before you run this script.

  PLEASE DO NOT COPY THIS SCRIPT TO ANY OTHER LOCATION. YOU WILL MISS IMPORTANT UPDATES.

  Required parameters for injecting files to be transferred:
  $0 --filename file --path path 
     --type type --config file [--destination default] [--filesize size] [--hostname host]
 
  Filename and path to file on the given host point to the file to be transferred.

  Type is the type of file, which requires extra parameters to be specified
  Current supported types: streamer, edm, lumi, dqm, pixdmp:
    - Streamers require runnumber, lumisection, numevents, appname, appversion, stream, 
      setuplabel, and index.
    - EDM files require runnumber, lumisection, numevents, appname, appversion and setuplabel.
      (EDM type can be choses also for general ROOT files)
    - Lumi files require runnumber, lumisection, appname, and appversion. 
    - DQM files  require runnumber, lumisection, appname, and appversion.
    - PIXDMP files require runnumber.

  Config file has to specify a user and password for the cms_rcms online database. At best
  see the example in ~tier0/.tier0trans.conf.

  Destination determines where the file goes on Tier0. It will be set to default if not set by 
  user (and typically there should be no need to change it).

  Filesize [in Bytes] is required. Since in most case submitting host is where the file resides 
  the filesize will be determined automatically.

  Hostname is the host on which the file is found. By default, this will be set to the name as 
  returned by the `hostname` command. Currently supported hosts for copies are: cms-tier0-stage, 
  cmsdisk1, csc-expert, vmepcS2B18-39 (tracker node), srv-C2D05-19 (DQM) and the Storage Manager nodes.
 
  --------------------------------------------------------------------------------------------
  If you are not sure about what you are doing please send an inquiry to hn-cms-tier0-operations\@cern.ch.
  --------------------------------------------------------------------------------------------

  Other parameters (leave out if you do not know what they mean):
  --debug           : Print out extra messages
  --test            : Run in test modus to check logic. No DB inserts or T0 notification.
  --renotify        : Use this option if your files are stuck in FILES_INJECTED
  --producer        : Producer of file
  --appname         : Application name for file (e.g. CMSSW)
  --appversion      : Application version (e.g. CMSSW_2_0_8)
  --runnumber       : Run number file belongs to
  --lumisection     : Lumisection of file
  --setuplabel      : Setup label used in the configuration
  --dataset         : Same as --setuplabel
  --count           : Count within lumisection
  --stream          : Stream file comes from
  --instance        : Instance of creating application
  --nevents         : Number of events in the file
  --ctime           : Creation time of file in seconds since epoch, defaults to current time
  --itime           : Injection time of file in seconds since epoch, set to current time
                    : (File times are for bookkeeping purposes; use your own time if desired)
  --index           : Name of index file (default is to changing data file .dat to .ind)
  --checksum        : Checksum of the file
  --comment         : Comment field in the database

  --------------------------------------------------------------------------------------------
  | IMPORTANT --- QUERY/CHECK mode --- IMPORTANT                            
  --------------------------------------------------------------------------------------------

  $0 --check --filename file
  Check on the status of a file previously injected into the transfer system.
  ############################################################################################  
  \n";
    exit 1;
}

# subroutine for getting formatted time for SQL to_date method
sub gettimestamp($) {
    my $stime = shift;
    my @ltime = localtime($stime);
    my ( $sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst ) = @ltime;

    $year += 1900;
    $mon++;

    my $timestr = $year . "-";
    if ( $mon < 10 ) {
        $timestr = $timestr . "0";
    }

    $timestr = $timestr . $mon . "-";

    if ( $mday < 10 ) {
        $timestr = $timestr . "0";
    }

    $timestr = $timestr . $mday . " " . $hour . ":" . $min . ":" . $sec;
    return $timestr;
}

# strip quotes and otherwise check that a parameter is in good format
sub checkOption($) {

    my $theOpt = shift;

    # strip off any double or single quotes
    # they won't play nice with sql query or notify script call
    $theOpt =~ s/\'//g;
    $theOpt =~ s/\"//g;

    # check string for spaces
    # return an error if found
    if ( $theOpt =~ / / ) {
        print
"Option specified as '$theOpt' has spaces in it. Please use without spaces. Exit\n";
        exit 1;
    }

    return $theOpt;
}

my (
    $help,  $debug,     $hostname, $filename, $pathname,
    $index, $indexsize, $filesize, $test
);
my ( $producer, $stream, $type, $runnumber, $lumisection, $count, $instance );
my ( $createtime, $injecttime, $ctime, $itime, $comment, $destination );
my ( $appname, $appversion, $nevents, $checksum, $setuplabel, $check, $config,
    $renotify );

$help        = 0;
$debug       = 0;
$test        = 0;
$hostname    = 'unset';
$filename    = '';
$pathname    = '';
$destination = '';
$filesize    = 0;

# these optional parameters must not be empty strings
# transfer system requires these options be set to SOMETHING, even if its meaningless
$producer    = 'default';
$stream      = '';
$type        = '';
$runnumber   = 0;
$lumisection = -1;
$count       = -1;
$instance    = -1;
$nevents     = 0;
$ctime       = 0;
$itime       = 0;
$appname     = '';
$appversion  = '';
$checksum    = '';
$setuplabel  = 'default';
$destination = 'default';
$index       = '';
$indexsize   = -1;
$comment     = '';
$config      = '';

GetOptions(
    "h|help"                   => \$help,
    "debug"                    => \$debug,
    "test"                     => \$test,
    "renotify"                 => \$renotify,
    "check"                    => \$check,
    "config=s"                 => \$config,
    "hostname=s"               => \$hostname,
    "file|filename=s"          => \$filename,
    "path|pathname=s"          => \$pathname,
    "destination=s"            => \$destination,
    "setuplabel|dataset=s"     => \$setuplabel,
    "filesize=i"               => \$filesize,
    "producer=s"               => \$producer,
    "run|runnumber=i"          => \$runnumber,
    "lumisection=i"            => \$lumisection,
    "count=i"                  => \$count,
    "instance=i"               => \$instance,
    "stream=s"                 => \$stream,
    "type=s"                   => \$type,
    "index=s"                  => \$index,
    "nevents|numevents=i"      => \$nevents,
    "appname|app_name=s"       => \$appname,
    "appversion|app_version=s" => \$appversion,
    "checksum=s"               => \$checksum,
    "ctime|start_time=i"       => \$ctime,
    "itime|stop_time=i"        => \$itime,
    "comment=s"                => \$comment
) or usage;

$help && usage;

# enable debug for test option
if ($test) { $debug = 1; }

############################################
# Main starts here
############################################

# first check formatting of options
$filename    = checkOption($filename);
$pathname    = checkOption($pathname);
$hostname    = checkOption($hostname);
$destination = checkOption($destination);
$setuplabel  = checkOption($setuplabel);
$producer    = checkOption($producer);
$stream      = checkOption($stream);
$type        = checkOption($type);
$index       = checkOption($index);
$appname     = checkOption($appname);
$appversion  = checkOption($appversion);
$checksum    = checkOption($checksum);

# explicit way to do comment since it can do ok with spaces
$comment =~ s/\'//g;
$comment =~ s/\"//g;

my $reader = "XXX";
my $phrase = "xxx";

if ( $config eq '' ) {
    print
      "Error: You have to specify a config file (--config option), exiting!\n";
    usageShort();
}
elsif ( -e $config ) {
    eval `cat $config`;
}
else {
    print "Error: Can not read config file: $config, exiting!\n";
    usageShort();
}

unless ($filename) {
    print "Error: No filename supplied, exiting!\n";
    usageShort();
}

# redirect signals
$SIG{ABRT} = \&IGNORE;
$SIG{INT}  = \&IGNORE;
$SIG{KILL} = \&IGNORE;
$SIG{QUIT} = \&IGNORE;
$SIG{TERM} = \&IGNORE;

# when $check is enabled, just want to query for a file and then exit
if ($check) {
    my $SQLcheck =
        "select"
      . "  CMS_STOMGR.FILES_CREATED.FILENAME"
      . ", CMS_STOMGR.FILES_INJECTED.FILENAME"
      . ", CMS_STOMGR.FILES_TRANS_NEW.FILENAME"
      . ", CMS_STOMGR.FILES_TRANS_COPIED.FILENAME"
      . ", CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME"
      . ", CMS_STOMGR.FILES_TRANS_INSERTED.FILENAME"
      . ", CMS_STOMGR.FILES_DELETED.FILENAME"
      . " from CMS_STOMGR.FILES_CREATED "
      . "left outer join CMS_STOMGR.FILES_INJECTED "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME "
      . "left outer join CMS_STOMGR.FILES_TRANS_NEW "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_NEW.FILENAME "
      . "left outer join CMS_STOMGR.FILES_TRANS_COPIED  "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_COPIED.FILENAME "
      . "left outer join CMS_STOMGR.FILES_TRANS_CHECKED "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME "
      . "left outer join CMS_STOMGR.FILES_TRANS_INSERTED "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_INSERTED.FILENAME "
      . "left outer join CMS_STOMGR.FILES_DELETED "
      . "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_DELETED.FILENAME "
      . "where CMS_STOMGR.FILES_CREATED.FILENAME=?";

    #
    sleep(3);

    #

    # setup DB connection
    my $dbi = "DBI:Oracle:cms_rcms";
    $debug && print "Setting up DB connection for $dbi and $reader\n";
    my $dbh = DBI->connect( $dbi, $reader, $phrase )
      or die("Error: Connection to Oracle DB failed");
    $debug && print "DB connection set up successfully \n";

    $debug && print "Preparing check query: $SQLcheck \n";
    my $checkHan = $dbh->prepare($SQLcheck)
      or die("Error: DB query prepare failed - $dbh->errstr \n");

    my $dbErr;
    $debug && print "Querying tables for file status \n";
    my $rowsCcheck = $checkHan->execute($filename)
      or $dbErr = $checkHan->errstr;
    if ( defined($dbErr) ) {
        print "Error: Query failed.\n";
        print "Error string from DB is $dbErr\n";
        exit 1;
    }

    #Get query result - array elements will be '' when file not in that table.
    my @result = $checkHan->fetchrow_array;
    unless ( $result[0] ) { print "File not found in database.\n"; exit; }
    unless ( $result[1] ) {
        print
"FILES_CREATED: File found in database but not passed over to T0 system.\n";
        exit 0;
    }
    unless ( $result[2] ) {
        print
"FILES_INJECTED: File found in database and handed over to T0 system.\n";
        exit 0;
    }
    unless ( $result[3] ) {
        print
"FILES_TRANS_NEW: File found in database and being processed by T0 system.\n";
        exit 0;
    }
    unless ( $result[4] ) {
        print
"FILES_TRANS_COPIED: File found in database and copied by T0 system.\n";
        exit 0;
    }
    unless ( $result[5] ) {
        print
"FILES_TRANS_CHECKED: File found in database and checked by T0 system.\n";
        exit 0;
    }
    unless ( $result[6] ) {
        print
"FILES_TRANS_INSERTED: File found in database and sucessfully processed by T0 system.\n";
        exit 0;
    }
    print
"FILES_DELETED: File found in database, sucessfully processed and locally deleted\n";
    exit 0;
}

# filename, path, host and filesize must be correct or transfer will never work
# first check to make sure all of these are set and exit if not
unless ($pathname) {
    print "Error: No path supplied, exiting!\n";
    usageShort();
}

# try to match hostname and alias
$hostname = hostname()     if $hostname eq 'unset';
$hostname = 'srv-C2D05-03' if $hostname eq 'cms-tier0-stage';
$hostname = 'srv-C2D05-02' if $hostname eq 'cmsdisk1';
$hostname = 'csc-C2D08-12' if $hostname eq 'csc-expert';

unless (
       $hostname eq 'srv-C2D05-03'
    || $hostname eq 'srv-C2D05-02'
    || $hostname eq 'csc-C2D08-12'
    ||    #csc
    $hostname eq 'vmepcS2B18-39' ||    #tracker
    $hostname eq 'srv-C2D05-19' ||     #dqm
    $hostname =~ 'srv-C2C06-' || $hostname =~ 'srv-C2C07-'
  )
{
    print
"Error: Hostname not valid. Must be one of cms-tier0-stage, cmsdisk1, csc-daq00, srv-C2D05-19 or vmepcS2B18-39.\n";
    usageShort();
}

# test if file is present
if ( hostname() eq $hostname && !( -e "$pathname/$filename" ) ) {
    print "Error: Hostname = this machine, but file does not exist, exiting!\n";
    usageShort();
}
elsif ( hostname() ne $hostname && ( -e "$pathname/$filename" ) ) {
    print
"Error: Hostname != this machine, but file exists on this host, exiting!\n";
    usageShort();
}

# test if file is world readable
if ( hostname() eq $hostname ) {
    system("chmod a+r $pathname/$filename");
    my $mode = ( stat("$pathname/$filename") )[2];
    $mode = $mode & 0x0007;
    if ( $mode < 4 ) {
        print
"Error: Hostname = this machine, but file is not readable by others, exiting!\n";
        usageShort();
    }
}

# if we are running this on same host as the file can find out filesize as a fallback
unless ($filesize) {
    print "Warning: No filesize supplied (or filesize=0 chosen), ";
    if ( hostname() ne $hostname ) {
        print "exiting!\n";
        usageShort();
    }
    else {
        print "but hostname = this machine: using size of file on disk \n";
        $filesize = -s "$pathname/$filename";
    }
}

unless ($type) {
    print "Error: No type specified, exiting!\n";
    usageShort();
}

# depending on type check for different required parameters
if ( $type eq "streamer" ) {
    unless ( $runnumber
        && $lumisection != -1
        && $nevents
        && $appname
        && $appversion
        && $stream
        && $setuplabel ne 'default' )
    {
        print
"Error: For streamer files need runnumber, lumisection, numevents, appname, appversion, stream, setuplabel, and index specified\n";
        usageShort();
    }
}
elsif ( $type eq "edm" ) {
    unless ( $runnumber
        && $lumisection != -1
        && $nevents
        && $appname
        && $appversion
        && $setuplabel ne 'default' )
    {
        print
"Error: For edm files need runnumber, lumisection, numevents, appname, appversion and setuplabel specified\n";
        usageShort();
    }
}
elsif (( $type eq "lumi" )
    || ( $type eq "lumi-sa" )
    || ( $type eq "lumi-vdm" ) )
{
    $destination = 'cms_lumi' if ( $destination eq 'default' );
    unless ( $runnumber && $lumisection != -1 && $appname && $appversion ) {
        print
"Error: For lumi files need runnumber, lumisection, appname, and appversion specified.\n";
        usageShort();
    }
}
elsif ( $type eq "dqm" ) {
    $destination = 'dqm' if ( $destination eq 'default' );
    unless ( $runnumber && $lumisection != -1 && $appname && $appversion ) {
        print
"Error: For dqm files need runnumber, lumisection, appname, and appversion specified.\n";
        usageShort();
    }
}
elsif ( $type eq "pixdmp" ) {
    $setuplabel  = 'PixelCalib' if ( $setuplabel  eq 'default' );
    $destination = 'pixdmp'     if ( $destination eq 'default' );
    unless ( $runnumber && $nevents ) {
        print
"Error: For pixdmp files need runnumber and numevents to be specified\n";
        usageShort();
    }
}
else {
    print "Error: File type not a recognized type.\n"
      . "Recognized types are streamer, edm, and lumi";
    usageShort();
}

( $destination eq 'default' )
  && $debug
  && print "No destination specified, default will be used \n";

# will just use time now as creation and injection if no other time specified.
if ( !$ctime ) {
    $ctime = time;
    $debug && print "No creation time specified, using current time \n";
}
$createtime = gettimestamp($ctime);

if ( !$itime ) {
    $itime = time;
    $debug && print "Injection time set to current time \n";
}
$injecttime = gettimestamp($itime);

# if file is a .dat and there is an index file, supply it
my $indfile = '';
if ( $filename =~ /\.dat$/ && !$index ) {
    $indfile = $filename;
    $indfile =~ s/\.dat$/\.ind/;
    if ( -e "$pathname/$indfile" ) {
        $index     = $indfile;
        $indexsize = -s "$pathname/$index";
    }
#    elsif ( $type eq 'streamer' ) {
#        print
#"Index file required for streamer files, not found in usual place please specify. Exiting!\n";
#        usage();
#        exit 1;
#    }
}

# create inserts into FILES_CREATED and FILES_INJECTED
my $SQLcquery =
"SELECT FILENAME,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION,"
  . " RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME"
  . " from CMS_STOMGR.FILES_CREATED where CMS_STOMGR.FILES_CREATED.FILENAME='$filename'";

my $SQLiquery =
    "SELECT FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME"
  . " from CMS_STOMGR.FILES_INJECTED where CMS_STOMGR.FILES_INJECTED.FILENAME='$filename'";

my $SQLcreate =
    "INSERT INTO CMS_STOMGR.FILES_CREATED ("
  . "FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION,"
  . "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME,COMMENT_STR) "
  . "VALUES ('$filename','$pathname','$hostname','$setuplabel','$stream','$type','$producer',"
  . "'$appname','$appversion',$runnumber,$lumisection,$count,$instance,"
  . "TO_DATE('$createtime','YYYY-MM-DD HH24:MI:SS'),'$comment')";

my $SQLinject =
    "INSERT INTO CMS_STOMGR.FILES_INJECTED ("
  . "FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,COMMENT_STR,INDFILENAME,INDFILESIZE) "
  . "VALUES ('$filename','$pathname','$destination',$nevents,$filesize,'$checksum',"
  . "TO_DATE('$injecttime','YYYY-MM-DD HH24:MI:SS'),'$comment','$index',$indexsize)";

$debug && print "SQL commands:\n $SQLcreate \n $SQLinject \n";

# notify script can be changed by environment variable
my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
if ( !defined $notscript ) {
    $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
}

# all these options are enforced by notify script (even if meaningless):
my $TIERZERO =
"$notscript --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --FILESIZE $filesize --TYPE $type "
  . "--START_TIME $ctime --STOP_TIME $itime --SETUPLABEL $setuplabel --DESTINATION $destination "
  . "--HLTKEY UNKNOWN";

# these options aren't needed but available:
if ($runnumber)           { $TIERZERO .= " --RUNNUMBER $runnumber"; }
if ( $lumisection != -1 ) { $TIERZERO .= " --LUMISECTION $lumisection"; }
if ($stream)              { $TIERZERO .= " --STREAM $stream"; }
if ($nevents)             { $TIERZERO .= " --NEVENTS $nevents"; }
if ($index) {
    $TIERZERO .= " --INDEX $index";

    #                        $TIERZERO .= " --INDEXFILESIZE $indfilesize";
}
if ($appname)  { $TIERZERO .= " --APP_NAME $appname"; }
if ($appname)  { $TIERZERO .= " --APP_VERSION $appversion"; }
if ($checksum) { $TIERZERO .= " --CHECKSUM $checksum"; }
$debug && print "Notify command: \n $TIERZERO \n";

#
sleep(1);

#

# do we just renotify
if ( defined $renotify ) {
    my $T0out = `$TIERZERO 2>&1`;
    if ( $T0out =~ /Connection established/ ) {
        print "File sucessfully re-submitted for transfer.\n";
        exit 0;
    }
    else {
        print
"Did not connect properly to transfer system logger. Error follows below\n\n";
        print $T0out;
        print "\n";
        exit 1;
    }
}

# setup DB connection
my $dbi = "DBI:Oracle:cms_rcms";
$debug && print "Setting up DB connection for $dbi and $reader\n";
my $dbh = DBI->connect( $dbi, $reader, $phrase )
  or die("Error: Connection to Oracle DB failed");
$debug && print "DB connection set up successfully \n";

# do DB checks
my $dodbins = 1;
my $dot0not = 1;

my $dbcheck = 1;
if ( $dbcheck == 1 ) {
    my $cqueryHan = $dbh->prepare($SQLcquery)
      or die("Error: Prepare failed, $dbh->errstr \n");
    my $dbcqErr;
    my $cqstr = $cqueryHan->execute() or $dbcqErr = $cqueryHan->errstr;
    if ( defined($dbcqErr) ) {
        print "Error: Check query on FILES_CREATED failed.\n";
        print "Error string from DB is $dbcqErr\n";
        $dbh->disconnect;
        exit 1;
    }

    my @cqres = $cqueryHan->fetchrow_array;
    $cqueryHan->finish;

    my @iqres;
    if ( defined( $cqres[0] ) ) {
        my $iqueryHan = $dbh->prepare($SQLiquery)
          or die("Error: Prepare failed, $dbh->errstr \n");
        my $dbiqErr;
        my $iqstr = $iqueryHan->execute() or $dbiqErr = $iqueryHan->errstr;
        if ( defined($dbiqErr) ) {
            print "Error: Check query on FILES_INJECTED failed.\n";
            print "Error string from DB is $dbiqErr\n";
            $dbh->disconnect;
            exit 1;
        }
        @iqres = $iqueryHan->fetchrow_array;
        $iqueryHan->finish;

        print
"Error: Found file in DB. To see its status please execute:\n $0 --check --filename $filename\n";
        exit 1;
        $dodbins = 0;
    }
}

# do DB inserts
$debug && print "Preparing DB inserts\n";
my $createHan = $dbh->prepare($SQLcreate)
  or die("Error: Prepare failed, $dbh->errstr \n");
my $injectHan = $dbh->prepare($SQLinject)
  or die("Error: Prepare failed, $dbh->errstr \n");

my $dbErr;
$debug && print "Inserting into FILES_CREATED \n";
my $rowsCreate = 0;
if ( $test == 0 ) {
    $rowsCreate = $createHan->execute() or $dbErr = $createHan->errstr;
    if ( defined($dbErr) ) {
        print "Error: Insert into FILES_CREATED failed.\n";
        if ( $dbErr =~ /unique constraint/ ) {
            print "File is already in the database.\n";
        }
        print "Error string from DB is $dbErr\n";
        $dbh->disconnect;
        exit 1;
    }
}

$debug && print "Inserting into FILES_INJECTED \n";
my $rowsInject = 0;
if ( $test == 0 ) {
    $rowsInject = $injectHan->execute() or $dbErr = $injectHan->errstr;
    if ( defined($dbErr) ) {
        print "Error: Insert into FILES_INJECTED failed.\n";
        if ( $dbErr =~ /unique constraint/ ) {
            print "File is already in the database.\n";
        }
        print "Error string from DB is $dbErr\n";
        $dbh->disconnect;
        exit 1;
    }
}

if ( $test == 0 ) {
    my $T0out;
    if ( $rowsCreate == 1 && $rowsInject == 1 ) {
        print "DB inserts completed, running Tier 0 notification script.\n";
        $T0out = `$TIERZERO 2>&1`;
        if ( $T0out =~ /Connection established/ ) {
            print "File sucessfully submitted for transfer.\n";
        }
        else {
            print
"Did not connect properly to transfer system logger. Error follows below\n\n";
            print $T0out;
            print "\n";
        }
    }
    else {
        ( $rowsCreate != 1 )
          && print
          "Error: DB returned strange code on $SQLcreate - rows=$rowsCreate\n";
        ( $rowsInject != 1 )
          && print
          "Error: DB returned strange code on $SQLinject - rows=$rowsInject\n";
        print "Not calling Tier 0 notification script\n";
    }
}

$dbh->disconnect;

if ( $test == 1 ) {
    print "\n\nNo obvious logic errors detected\n";
}

# reset signal
$SIG{ABRT} = 'DEFAULT';
$SIG{INT}  = 'DEFAULT';
$SIG{KILL} = 'DEFAULT';
$SIG{QUIT} = 'DEFAULT';
$SIG{TERM} = 'DEFAULT';

exit 0;
