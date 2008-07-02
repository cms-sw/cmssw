#!/usr/bin/perl -w
# $Id: injectFileIntoTransferSystem.pl,v 1.12 2008/07/02 13:32:45 jserrano Exp $
#
# Written by Matt Rudolph June 2008
#

use strict;
use DBI;
use Getopt::Long;
use File::Basename;
use Sys::Hostname;


sub usageShort
{
  print "
  ############################################################################################
  For more information please see help:
  $0 --help
  ############################################################################################
  \n";
  exit;
}

sub usage
{
  print " 
  ############################################################################################
  Usage:
  $0 --help  : show this message

  --------------------------------------------------------------------------------------------

  Required parameters for injecting files to be transferred:
  $0 --filename file --path path --filesize size 
                                    --type type --hostname host [--destination default]
 
  Filename, path, and filesize are self explanatory.

  Type is the type of file, which requires extra parameters to be specified
  Current supported types: streamer, edm, lumi:
    Streamers require runnumber, lumisection, numevents, appname, appversion, stream, 
      setup abel, and index specified
    Edm files require runnumber, lumisection, numevents, appname, appversion, setuplabel, stream
    Lumi files require runnumber, lumisection, appname, and appversion

  Hostname is the host on which the file is found. (This should be the name as returned by the
  `hostname` command. Supported hosts for copies: cmsdisk1, srv-c2c06-02.cms (cmsmon) and
  the Storage Manager nodes.

  Destination determines where file goes on Tier0. It is set to default if not set by user.
 
  If you are not sure about what you are doing please send an inquiry to hn-tier0-ops\@cern.ch.

  Other parameters:
  --debug           : Print out extra messages
  --debug           : Print out extra messages
  --producer        : Producer of file
  --appname         : Application name for file (e.g. CMSSW)
  --appversion      : Application version
  --runnumber       : Run number file belongs to
  --lumisection     : Lumisection of file
  --count           : Count within lumisection
  --stream          : Stream file comes from
  --instance        : Instance of creating application
  --nevents         : Number of events in the file
  --ctime           : Creation time of file in seconds since epoch, defaults to current time
  --itime           : Injection time of file in seconds since epoch, set to current time
                    : File times are for bookkeeping purposes; use own time if desired
  --index           : Name of index file defaults to changing data file .dat to .ind
  --checksum        : Checksum of the file
  --comment         : Comment field in the database

  --------------------------------------------------------------------------------------------

  $0 --check --filename=file
  Check on the status of a file previously injected into the transfer system.
  ############################################################################################  
  \n";
  exit;
}

# subroutine for getting formatted time for SQL to_date method
sub gettimestamp($)
{
    my $stime = shift;
    my @ltime = localtime($stime);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;

    $year += 1900;
    $mon++;


    my $timestr = $year."-";
    if ($mon < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mon . "-";

    if ($mday < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mday . " " . $hour . ":" . $min . ":" . $sec;
    return $timestr;
}

# strip quotes and otherwise check that a parameter is in good format
sub checkOption($) {

    my $theOpt=shift;

    # strip off any double or single quotes 
    # they won't play nice with sql query or notify script call
    $theOpt =~ s/\'//g;
    $theOpt =~ s/\"//g;

    # check string for spaces
    # return an error if found
    if($theOpt=~ / /) {
	print "Option specified as '$theOpt' has spaces in it.  Please use without spaces.  Exit\n";
	exit;
    }

    return $theOpt;
}

my ($help, $debug, $hostname, $filename, $pathname, $index, $filesize);
my ($producer, $stream, $type, $runnumber, $lumisection, $count,$instance);
my ($createtime, $injecttime, $ctime, $itime, $comment, $destination);
my ($appname, $appversion, $nevents, $checksum, $setuplabel, $check);

$help        = 0;
$debug       = 0;
$hostname    = '';
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
$comment     = '';

GetOptions(
           "h|help"                   => \$help,
           "debug"                    => \$debug,
           "check"                    => \$check,
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

unless($filename) {
    print "Error: No filename supplied, exiting \n";
    usageShort();
}

# when $check is enabled, just want to query for a file and then exit
if($check) {
    my $SQLcheck = "select CMS_STOMGR.FILES_CREATED.FILENAME".
	", CMS_STOMGR.FILES_INJECTED.FILENAME".
	", CMS_STOMGR.FILES_TRANS_NEW.FILENAME".
        ", CMS_STOMGR.FILES_TRANS_COPIED.FILENAME".
	", CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME".
	", CMS_STOMGR.FILES_TRANS_INSERTED.FILENAME".
	", CMS_STOMGR.FILES_DELETED.FILENAME".
	" from CMS_STOMGR.FILES_CREATED " .
        "left outer join CMS_STOMGR.FILES_INJECTED ".
	"on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_TRANS_NEW ".
	"on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_NEW.FILENAME " .
        "left outer join CMS_STOMGR.FILES_TRANS_COPIED  ".
	"on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_COPIED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_TRANS_CHECKED ".
	"on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_TRANS_INSERTED ".
        "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_INSERTED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_DELETED ".
        "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_DELETED.FILENAME " .
        "where CMS_STOMGR.FILES_CREATED.FILENAME='$filename'";

    # setup DB connection
    my $dbi    = "DBI:Oracle:cms_rcms";
    my $reader = "CMS_STOMGR_W";
    $debug && print "Setting up DB connection for $dbi and $reader\n";
    my $dbh = DBI->connect($dbi,$reader,"qwerty") or die("Error: Connection to Oracle DB failed");
    $debug && print "DB connection set up succesfully \n";

    $debug && print "Preparing check query: $SQLcheck \n";
    my $checkHan = $dbh->prepare($SQLcheck) or die("Error: DB query prepare failed - $dbh->errstr \n");
    
    my $dbErr;
    $debug && print "Querying tables for file status \n";
    my $rowsCcheck = $checkHan->execute() or $dbErr=$checkHan->errstr;
    if(defined($dbErr)) {
	print "Error: Query failed.\n";
	print "Error string from db was $dbErr\n";
	exit;
    }

    #Get query result - array elements will be '' when file not in that table.
    my @result = $checkHan->fetchrow_array;

    unless($result[0]) {print "File not found in database.\n"; exit;}
    unless($result[1]) {print "FILES_CREATED\n"; exit;}
    unless($result[2]) {print "FILES_INJECTED\n"; exit;}
    unless($result[3]) {print "FILES_TRANS_NEW\n"; exit;}
    unless($result[4]) {print "FILES_TRANS_COPIED\n"; exit;}
    unless($result[5]) {print "FILES_TRANS_CHECKED\n"; exit;}
    unless($result[6]) {print "FILES_TRANS_INSERTED\n"; exit;}
    print "FILES_DELETED.\n"; exit;
}

# filename, path, host and filesize must be correct or transfer will never work
# first check to make sure all of these are set and exit if not



unless($pathname) {
    print "Error: No path supplied, exiting \n";
    usageShort();
}

unless($hostname eq 'cmsdisk1' || $hostname eq 'srv-c2c06-02.cms' || $hostname =~ 'srv-c2c07-' || $hostname =~ 'srv-C2C07-') {
    print "Error: Hostname not valid.  Must be cmsdisk1, srv-c2c06-02.cms, or a Storage Manager node\n";
    usageShort();
}

if(hostname() eq $hostname && !(-e "$pathname/$filename")) {
    print "Error: Hostname = this machine, but file does not exist, exiting \n";
    usageShort();
} 

# if we are running this on same host as the file can find out filesize as a fallback
unless($filesize) {
    print "Warning: No filesize supplied or filesize=0, ";
    if(hostname() ne $hostname) {
	print "exiting \n";
        usageShort();
    } else {
	print "but hostname = this machine: using size of file on disk \n"; 
	$filesize = -s "$pathname/$filename";
    }
}

unless($type) {
    print "Error: No type specified, exiting \n";
    usageShort();
}

# depending on type check for different required parameters
if($type eq "streamer") {
    unless( $runnumber && $lumisection != -1 && $nevents && $appname && $appversion && $stream && $setuplabel ne 'default') {
	print "Error: For streamer files need runnumber, lumisection, numevents, appname, appversion, stream, setuplabel, and index specified\n";
        usageShort();
    }
} elsif($type eq "edm") {
    unless($runnumber && $lumisection != -1 && $nevents && $appname && $appversion && $stream && $setuplabel ne 'default') {
	print "Error: For edm files need runnumber, lumisection, numevents, appname, appversion, setuplabel, and stream specified\n";
        usageShort();
    }
} elsif($type eq "lumi") {
    $destination = 'cms_lumi' if ($destination eq 'default');
    unless( $runnumber && $lumisection != -1 && $appname && $appversion) {
	print "Error: For lumi files need runnumber, lumisection, appname, and appversion specified.\n";
        usageShort();
    }
} else {
    print "Error: File type not a recognized type.\n" .
          "Recognized types are streamer, edm, and lumi";
    usageShort();
}

($destination eq 'default') && $debug && print "No destination specified, default will be used \n";

# will just use time now as creation and injection if no other time specified.
if(!$ctime) {
    $ctime =time;
    $debug && print "No creation time specified, using time now \n";
}
$createtime = gettimestamp($ctime);

if(!$itime) {
    $itime=time;
    $debug && print "Injection time set to current time \n";
}
$injecttime = gettimestamp($itime);

# create inserts into FILES_CREATED and FILES_INJECTED
my $SQLcreate = "INSERT INTO CMS_STOMGR.FILES_CREATED (" .
            "FILENAME,CPATH,HOSTNAME,SETUPLABEL,STREAM,TYPE,PRODUCER,APP_NAME,APP_VERSION," .
            "RUNNUMBER,LUMISECTION,COUNT,INSTANCE,CTIME,COMMENT_STR) " .
            "VALUES ('$filename','$pathname','$hostname','$setuplabel','$stream','$type','$producer'," .
            "'$appname','$appversion',$runnumber,$lumisection,$count,$instance," .
            "TO_DATE('$createtime','YYYY-MM-DD HH24:MI:SS'),'$comment')";
my $SQLinject = "INSERT INTO CMS_STOMGR.FILES_INJECTED (" .
               "FILENAME,PATHNAME,DESTINATION,NEVENTS,FILESIZE,CHECKSUM,ITIME,COMMENT_STR) " .
               "VALUES ('$filename','$pathname','$destination',$nevents,$filesize,'$checksum'," . 
               "TO_DATE('$injecttime','YYYY-MM-DD HH24:MI:SS'),'$comment')";

$debug && print "SQL commands:\n $SQLcreate \n $SQLinject \n";

# notify script can be changed by environment variable
my $notscript = $ENV{'SM_NOTIFYSCRIPT'};
if (!defined $notscript) {
    $notscript = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh";
}


# if file is a .dat and there is an index file, supply it
my $indfile='';
if($filename=~/\.dat$/  && !$index) {
    $indfile=$filename;
    $indfile =~ s/\.dat$/\.ind/;
    if (-e "$pathname/$indfile") { 
	$index = $indfile;
    } elsif($type eq 'streamer') {
	print "Index file required for streamer files, not found in usual place please specify. Exiting \n";
        usage();
	exit;
    }
}

# all these options are enforced by notify script (even if meaningless):
my $TIERZERO = "$notscript --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --FILESIZE $filesize --TYPE $type " . 
                          "--START_TIME $ctime --STOP_TIME $itime --SETUPLABEL $setuplabel --DESTINATION $destination";

# these options aren't needed but available:
if($runnumber)         {$TIERZERO .= " --RUNNUMBER $runnumber";}
if($lumisection != -1) {$TIERZERO .= " --LUMISECTION $lumisection";}
if($stream)            {$TIERZERO .= " --STREAM $stream";}
if($nevents)           {$TIERZERO .= " --NEVENTS $nevents";}
if($index)             {$TIERZERO .= " --INDEX $index";}
if($appname)           {$TIERZERO .= " --APP_NAME $appname";}
if($appname)           {$TIERZERO .= " --APP_VERSION $appversion";}
if($checksum)          {$TIERZERO .= " --CHECKSUM $checksum";}
$debug && print "Notify command: \n $TIERZERO \n";

# setup DB connection
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
$debug && print "Setting up DB connection for $dbi and $reader\n";
my $dbh = DBI->connect($dbi,$reader,"qwerty") or die("Error: Connection to Oracle DB failed");
$debug && print "DB connection set up succesfully \n";

# do DB inserts
$debug && print "Preparing DB inserts\n";
my $createHan = $dbh->prepare($SQLcreate) or die("Error: Prepare failed, $dbh->errstr \n");
my $injectHan = $dbh->prepare($SQLinject) or die("Error: Prepare failed, $dbh->errstr \n");

my $dbErr;
$debug && print "Inserting into FILES_CREATED \n";
my $rowsCreate = $createHan->execute() or $dbErr=$createHan->errstr;
if(defined($dbErr)) {
    print "Error: Insert into FILES_CREATED failed.\n";
    if($dbErr =~ /unique constraint/ ) { print "File was already in the database.\n";}
    print "Error string from db was $dbErr\n";
    exit;
}

$debug && print "Inserting into FILES_INJECTED \n";
my $rowsInject = $injectHan->execute() or $dbErr=$injectHan->errstr;
if(defined($dbErr)) {
    print "Error: Insert into FILES_INJECTED failed.\n";
    if($dbErr =~ /unique constraint/ ) { print "File was already in the database.\n";}
    print "Error string from db was $dbErr\n";
    exit;
}

my $T0out;

# check return values before calling notification script.  Want to make sure no DB errors
if($rowsCreate==1 && $rowsInject==1) {
    print "DB inserts completed, running Tier 0 notification script \n";
    $T0out=`$TIERZERO 2>&1`;
    if($T0out =~ /Connection established/ ) {
	print "File sucessfully submitted for transfer.\n";
    } else {
	print "Did not connect properly to transfer system logger. Error follows below\n\n";
	print $T0out;
	print "\n";
    }
} else {
    ($rowsCreate!=1) && print "Error: DB returned strange code on $SQLcreate - rows=$rowsCreate\n";
    ($rowsInject!=1) && print "Error: DB returned strange code on $SQLinject - rows=$rowsInject\n";
    print "Not calling Tier 0 notification script\n";
}

$dbh->disconnect;
