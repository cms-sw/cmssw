#!/usr/bin/env perl

use warnings;
use DBI;
use Getopt::Long;
use File::Basename;

use constant MIN_CLEAN_AGE => 7; #won't delete anything under X days

#Example Call:
#./cleanupOrphans.pl --max=10 --closedfiles --clean --age=30 --FILES_ALL --contains=Transfer --host=srv-C2C07-15

#get host running from
my $runHost = `hostname`;
chomp $runHost;

#Get the user
my $user = `whoami`;
chomp $user;

#get current time info
my @ltime = localtime(time);
my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;
$year+= 1900;
$mon++;

#config file
my $configFile = "/nfshome0/smpro/sm_scripts_cvs/operations/cleanupOrphans.cfg";
#parses configuration file
open CONFIG, $configFile or die "Can't open config file: $configFile:$!\n";
while (<CONFIG>) {
    chomp;                  # no newline
    s/#.*//;                # no comments
    s/^\s+//;               # no leading white
    s/\s+$//;               # no trailing white
    next unless length;     # anything left?
    my ($var, $value) = split(/\s*=\s*/, $_, 2);
    $Con{$var} = $value;
} 
close CONFIG;

#load host list
if (! -e $Con{"HOSTLIST"} ){
    die "ERROR: File $Con{'HOSTLIST'} missing!";
}

my $DO = "";
my $DO_HOST = "";
my $DO_REPORT="";
my $DO_FULL_REPORT="";
my $DO_CLEAN="";
my $DO_OPEN="";
my $DO_CLOSED="";
my $DO_FILES="";
my $DO_FILES_EMU="";
my $DO_FILE_MAX=99999999;
my $DO_FILE_AGE=0;
my $DO_FILE_CONTAINS="";
my $DO_FILES_CREATED="";
my $DO_FILES_INJECTED="";

my $DO_FILES_TRANS_NEW="";
my $DO_FILES_TRANS_COPIED="";
my $DO_FILES_TRANS_CHECKED="";
my $DO_FILES_TRANS_INSERTED="";
my $DO_FILES_DELETED="";
my $DO_FILES_NO_STATUS="";

#Set up Database Stuff
my $reader = "XXX";
my $phrase = "xxx";

$dbconfig = $Con{"DB_CONFIG"};
if (-e $dbconfig){
    eval `sudo -u smpro cat $dbconfig`;
}
else{
    die("Error: Unable to access database user information");
}

#Connect to the database
my $dbi = "DBI:Oracle:cms_rcms";
my $dbh = DBI->connect($dbi,$reader,$phrase) or die("Error: Connection to Oracle DB failed");

#Skeleton query for checking the status of a file
my $SQLcheck = "select" .
        "  CMS_STOMGR.FILES_CREATED.FILENAME".
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
        "where CMS_STOMGR.FILES_CREATED.FILENAME=?";
my $checkHan = $dbh->prepare($SQLcheck) or die("Error: DB query prepare failed - $dbh->errstr \n");

#Skeleton query to make entry into orphans table
my $SQLEntry = "insert into CMS_STOMGR.FILES_ORPHANS(FILENAME,DTIME,HOST,STATUS,COMMENT_STR) VALUES (?,sysdate,?,?,?)";
my $entryHan = $dbh->prepare($SQLEntry) or die("Error: DB query prepare failed - $dbh->errstr\n");

#Routine to get the status of a give file (based on previously existing separate script
sub getStatus($)
{
    my $filename = shift;
    my $rowsCcheck = $checkHan->execute($filename) or die("Error: Query failed - $dbh->errstr \n");
    
    my @result = $checkHan->fetchrow_array;
    unless($result[0]) {return -1;}
    unless($result[1]) {return 0;}
    unless($result[2]) {return 1;}
    unless($result[3]) {return 10;}
    unless($result[4]) {return 20;}
    unless($result[5]) {return 30;}
    unless($result[6]) {return 40;}
    return 99;
}   

#Generate a report (but not delete)
sub report($)
{
    my $path = shift;

    my $TOTAL_FILES_DONE = 0;
    my $TOTAL_FILES_CREATED = 0;
    my $TOTAL_FILES_INJECTED = 0;
    my $TOTAL_FILES_TRANS_NEW = 0;
    my $TOTAL_FILES_TRANS_COPIED = 0;
    my $TOTAL_FILES_TRANS_CHECKED = 0;
    my $TOTAL_FILES_TRANS_INSERTED = 0;
    my $TOTAL_FILES_DELETED = 0;
    my $TOTAL_FILES_NO_STATUS = 0;
    
    #If we want to restrict to a minimum age
    my $AGE_QUERY = "";
    if ($DO_FILE_AGE > 0) { $AGE_QUERY = "-mtime +$DO_FILE_AGE"; }

    #Keyword
    my $CONTAINS_QUERY = "";
    if ($DO_FILE_CONTAINS){ $CONTAINS_QUERY = "-name '*$DO_FILE_CONTAINS*'";}
    
    my @hosts = `cat $Con{"HOSTLIST"}`;
    if ($DO_HOST){ @hosts = $DO_HOST; }
    foreach $host (@hosts ){
	my $FILES_DONE = 0;
	my $FILES_CREATED = 0;
	my $FILES_INJECTED = 0;
	my $FILES_TRANS_NEW = 0;
	my $FILES_TRANS_COPIED = 0;
	my $FILES_TRANS_CHECKED = 0;
	my $FILES_TRANS_INSERTED = 0;
	my $FILES_DELETED = 0;
	my $FILES_NO_STATUS = 0;
	
	chomp $host;
	print "----- $host -----\n";
	my @files;
	if ($host eq $runHost) {@files = `find $path $AGE_QUERY $CONTAINS_QUERY -name '*.dat'`;}
	else {@files = `ssh $host "find $path $AGE_QUERY $CONTAINS_QUERY -name '*.dat'"`;}
	foreach $file (@files){
	    chomp $file;
	    if ($FILES_DONE < $DO_FILE_MAX){
		my $basename = `basename $file`;
		chomp $basename;
		my $status = &getStatus($basename);
		if ($DO_FULL_REPORT){
		    my $filedata;
		    if ($host eq $runHost) {$filedata = `find $file -printf '%TD, %kK'`;}
		    else {$filedata = `ssh $host "find $file -printf '%TD, %kK' "`;}
		    print "$host, $filedata, $status, $file\n";
		}
		if ($status == 0) {$FILES_CREATED = $FILES_CREATED + 1;}
		elsif ($status == 1) {$FILES_INJECTED = $FILES_INJECTED + 1;}
		elsif ($status == 10) {$FILES_TRANS_NEW = $FILES_TRANS_NEW + 1;}
		elsif ($status == 20) {$FILES_TRANS_COPIED = $FILES_TRANS_COPIED + 1;}
		elsif ($status == 30) {$FILES_TRANS_CHECKED = $FILES_TRANS_CHECKED + 1;}
		elsif ($status == 40) {$FILES_TRANS_INSERTED = $FILES_TRANS_INSERTED + 1;}
		elsif ($status == 99) {$FILES_DELETED = $FILES_DELETED + 1;}
		else {
		    $FILES_NO_STATUS = $FILES_NO_STATUS + 1;
		    #print "Internal error: unknown inject status\n";
		}
		$FILES_DONE = $FILES_DONE + 1;	      
	    }
	}
	
        $TOTAL_FILES_DONE = $TOTAL_FILES_DONE + $FILES_DONE;
        $TOTAL_FILES_CREATED = $TOTAL_FILES_CREATED + $FILES_CREATED;
        $TOTAL_FILES_INJECTED = $TOTAL_FILES_INJECTED + $FILES_INJECTED;
        $TOTAL_FILES_TRANS_NEW = $TOTAL_FILES_TRANS_NEW + $FILES_TRANS_NEW;
        $TOTAL_FILES_TRANS_COPIED = $TOTAL_FILES_TRANS_COPIED + $FILES_TRANS_COPIED;
        $TOTAL_FILES_TRANS_CHECKED = $TOTAL_FILES_TRANS_CHECKED + $FILES_TRANS_CHECKED;
        $TOTAL_FILES_TRANS_INSERTED = $TOTAL_FILES_TRANS_INSERTED + $FILES_TRANS_INSERTED;
        $TOTAL_FILES_DELETED = $TOTAL_FILES_DELETED + $FILES_DELETED;
	$TOTAL_FILES_NO_STATUS = $TOTAL_FILES_NO_STATUS + $FILES_NO_STATUS;

	print "\nDirectory Size\n";
	my $hostSizes;
	if ($host eq $runHost) {$hostSizes = `du -h $path`;}
	else {$hostSizes = `ssh $host "du -h $path"` ;}
	print "$hostSizes";
	print "FILE CATEGORIES ON DISK: \n";
	print "FILES_CREATED $FILES_CREATED \n";
	print "FILES_INJECTED $FILES_INJECTED \n";
	print "FILES_TRANS_NEW $FILES_TRANS_NEW \n";
	print "FILES_TRANS_COPIED $FILES_TRANS_COPIED \n";
	print "FILES_TRANS_CHECKED $FILES_TRANS_CHECKED \n";
	print "FILES_TRANS_INSERTED $FILES_TRANS_INSERTED \n";
	print "FILES_DELETED $FILES_DELETED \n";
	print "FILES_NO_STATUS_IN_DB $FILES_NO_STATUS \n";
	print "^^^^^^ $host ^^^^^^\n\n";	
    }
	

}

#Actually delete files
sub clean($)
{
    my $path = shift;
    
    my $TOTAL_FILES_DONE = 0;
    my $TOTAL_FILES_CREATED = 0;
    my $TOTAL_FILES_INJECTED = 0;
    my $TOTAL_FILES_TRANS_NEW = 0;
    my $TOTAL_FILES_TRANS_COPIED = 0;
    my $TOTAL_FILES_TRANS_CHECKED = 0;
    my $TOTAL_FILES_TRANS_INSERTED = 0;
    my $TOTAL_FILES_DELETED = 0;
    my $TOTAL_FILES_NO_STATUS = 0;

    #Open log file (specific to user and month)
    my $logfile = "/tmp/cleanupOrphans-$year-$mon-$user.log"; 
    open APPLOG, ">>$logfile";
    print APPLOG "\n-------------------- $mday-$mon-$year $hour:$min:$sec --------------------\n";
    
    #Adjust file age param if less than minimum for deletes
    my $tempDO_FILE_AGE = $DO_FILE_AGE;
    if ($DO_FILE_AGE < MIN_CLEAN_AGE){ 
	$DO_FILE_AGE = MIN_CLEAN_AGE;
	print "Age param too small for clean: setting to $DO_FILE_AGE.\n";
    }

    my $AGE_QUERY = "";
    if ($DO_FILE_AGE > 0) { $AGE_QUERY = "-mtime +$DO_FILE_AGE"; }

    my $CONTAINS_QUERY = "";
    if ($DO_FILE_CONTAINS){ $CONTAINS_QUERY = "-name '*$DO_FILE_CONTAINS*'";}
    
    my @hosts = `cat $Con{"HOSTLIST"}`;
    if ($DO_HOST){ @hosts = $DO_HOST; }
    foreach $host (@hosts ){
	my $FILES_DONE = 0;
	my $FILES_CREATED = 0;
	my $FILES_INJECTED = 0;
	my $FILES_TRANS_NEW = 0;
	my $FILES_TRANS_COPIED = 0;
	my $FILES_TRANS_CHECKED = 0;
	my $FILES_TRANS_INSERTED = 0;
	my $FILES_DELETED = 0;
	my $FILES_NO_STATUS = 0;
	my $COMPLETED_DELETES = 0;
	my $FAILED_DELETES = 0;
	
	chomp $host;
	print "----- $host -----\n";
	print APPLOG "    ----- $host -----\n";
	
	my @files;
	if ($host eq $runHost) {@files = `find $path $AGE_QUERY $CONTAINS_QUERY -name '*.dat'`;}
	else {@files = `ssh $host "find $path $AGE_QUERY $CONTAINS_QUERY -name '*.dat'"`;}
	foreach $file (@files){
	    chomp $file;
	    if ($FILES_DONE < $DO_FILE_MAX){
		my $basename = `basename $file`;
		chomp $basename;
		my $status = &getStatus($basename);
		if ($status == 0) {
		    if ($DO_FILES_CREATED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_CREATED = $FILES_CREATED + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 1) {
		    if ($DO_FILES_INJECTED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_INJECTED = $FILES_INJECTED + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 10) {
		    if ($DO_FILES_TRANS_NEW){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_TRANS_NEW = $FILES_TRANS_NEW + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 20) {
		    if ($DO_FILES_TRANS_COPIED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_TRANS_COPIED = $FILES_TRANS_COPIED + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 30) {
		    if ($DO_FILES_TRANS_CHECKED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_TRANS_CHECKED = $FILES_TRANS_CHECKED + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 40) {
		    if ($DO_FILES_TRANS_INSERTED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_TRANS_INSERTED = $FILES_TRANS_INSERTED = 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		elsif ($status == 99) {
		    if ($DO_FILES_DELETED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_DELETED = $FILES_DELETED + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}
		else {	   
		    #print "Internal error: unknown inject status\n";
		    if ($DO_FILES_NO_STATUS){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_NO_STATUS = $FILES_NO_STATUS + 1;
			$COMPLETED_DELETES = $COMPLETED_DELETES + 1;
			$FILES_DONE = $FILES_DONE + 1;
		    }
		}	      
	    } 
	}
	$COMPLETED_DELETES = $COMPLETED_DELETES - $FAILED_DELETES;

	$TOTAL_FILES_DONE = $TOTAL_FILES_DONE + $FILES_DONE;
        $TOTAL_FILES_CREATED = $TOTAL_FILES_CREATED + $FILES_CREATED;
        $TOTAL_FILES_INJECTED = $TOTAL_FILES_INJECTED + $FILES_INJECTED;
        $TOTAL_FILES_TRANS_NEW = $TOTAL_FILES_TRANS_NEW + $FILES_TRANS_NEW;
        $TOTAL_FILES_TRANS_COPIED = $TOTAL_FILES_TRANS_COPIED + $FILES_TRANS_COPIED;
        $TOTAL_FILES_TRANS_CHECKED = $TOTAL_FILES_TRANS_CHECKED + $FILES_TRANS_CHECKED;
        $TOTAL_FILES_TRANS_INSERTED = $TOTAL_FILES_TRANS_INSERTED + $FILES_TRANS_INSERTED;
        $TOTAL_FILES_DELETED = $TOTAL_FILES_DELETED + $FILES_DELETED;
	$TOTAL_FILES_NO_STATUS = $TOTAL_FILES_NO_STATUS + $FILES_NO_STATUS;
	
	print "FILES ON DISK: $FILES_DONE\n";
	print "\t FILES_CREATED $FILES_CREATED\n";
	print "\t FILES_INJECTED $FILES_INJECTED\n";
	print "\t FILES_TRANS_NEW $FILES_TRANS_NEW\n";
	print "\t FILES_TRANS_COPIED $FILES_TRANS_COPIED\n";
	print "\t FILES_TRANS_CHECKED $FILES_TRANS_CHECKED\n";
	print "\t FILES_TRANS_INSERTED $FILES_TRANS_INSERTED\n";
	print "\t FILES_DELETED $FILES_DELETED\n";
	if ($DO_FILES_NO_STATUS){
	    print "\t FILES_NO_STATUS_IN_DB $FILES_NO_STATUS\n"; 
	}
	else {
	    print "\t FILES_NO_STATUS_IN_DB(NOT DELETED) $FILES_NO_STATUS\n";
	}
	print "SUCCESSFUL DELETES: $COMPLETED_DELETES\n";
	print "FAILED DELETES: $FAILED_DELETES\n";
	print "Logfile: $logfile\n";
	
	print APPLOG "        [$mday-$mon-$year $hour:$min:$sec $host]FILES ON DISK: $FILES_DONE\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_CREATED $FILES_CREATED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_INJECTED $FILES_INJECTED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_TRANS_NEW $FILES_TRANS_NEW\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_TRANS_COPIED $FILES_TRANS_COPIED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_TRANS_CHECKED $FILES_TRANS_CHECKED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_TRANS_INSERTED $FILES_TRANS_INSERTED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_DELETED $FILES_DELETED\n";
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_NO_STATUS_IN_DB $FILES_NO_STATUS\n";
	print APPLOG "        [$mday-$mon-$year $hour:$min:$sec $host]SUCCESSFUL DELETES: $COMPLETED_DELETES\n";
	print APPLOG "        [$mday-$mon-$year $hour:$min:$sec $host]FAILED DELETES: $FAILED_DELETES\n";
    }
    close APPLOG;

    #reset the age param to original value
    $DO_FILE_AGE = $tempDO_FILE_AGE;
}

sub delete_file($)
{
    my $targetHost = shift;
    my $targetFile = shift;
    my $filename = shift;
    my $targetStatus = shift;
    substr($targetFile, index($targetFile, '.dat'), 4) = '.*';
    my $chmodexitcode = 0;
    my $rmexitcode = 0;
    my $numberFailed = 0;
    if ($DO_FILES_EMU){ #Right now this does this same thing but lists diagnostics
	if ($targetHost eq $runHost) {
	    $chmodexitcode = system("sudo chmod 666 $targetFile");
	    $rmexitcode = system("rm -f $targetFile");
	}
	else {
	    $chmodexitcode = system("ssh $targetHost 'sudo chmod 666 $targetFile'"); 
	    $rmexitcode = system("ssh $targetHost 'rm -f $targetFile'");
	}
	if ($rmexitcode != 0){
	    print "Failed: $filename\n";
	    $numberFailed = $numberFailed + 1;
	}
	else{
	    print "Removed: $filename\n";
	}
    }
    else { #No diagnostics
	if ($targetHost eq $runHost) {
	    $chmodexitcode = system("sudo chmod 666 $targetFile");
	    $rmexitcode = system("rm -f $targetFile");
	}
	else {
	    $chmodexitcode = system("ssh $targetHost 'sudo chmod 666 $targetFile'"); 
	    $rmexitcode = system("ssh $targetHost 'rm -f $targetFile'");
	}
	if ($rmexitcode != 0){
	    $numberFailed = $numberFailed + 1;
	}
    }

    if ($rmexitcode == 0){ #successful delete, so enter in DB
	if ($targetStatus > -1){ #Can't enter in table if not already in DB
	    $entryHan->bind_param(1,$filename);
	    $entryHan->bind_param(2,$host);
	    $entryHan->bind_param(3,$targetStatus);
	    $entryHan->bind_param(4,'');
	    $entryHan->execute();
	    my $entryErr = $entryHan->errstr;
	    if (defined($entryErr)){ #failed to enter in DB
		#Maybe exit, maybe log entry???
		print "Orphan DB insert produced error: $entryErr \n";
	    }
	}
    }
    return $numberFailed;
}

sub show_usage($)
{
    my $message = "$0 usage:\n\n".
	"   -h                     print this help\n".
	"   --help                 print this help\n\n".
	"Actions:\n".
	"   --report               print report\n".
	"   --fullreport           print report with file names\n".
	"   --clean                clean files\n\n".
	"Options:\n".
	"   --age=X                act only in files older than X days\n".
	"   --max=X                act only on up to X files per node\n".
	"   --host=hostname        provide hostname to run\n\n".
	"File Types:\n".
	"Actions only applied to selected file type.  You must select one file ype at least\n\n".
	"   --closedfiles          process closed files\n".
	"   --openfiles            process open files\n\n".
	"File Status:\n".
	"Only selected file status will be cleaned.  You must select one file status at least.\n\n".
	"   --FILES_CREATED\n".
	"   --FILES_INJECTED\n".
	"   --FILES_TRANS_NEW\n".
	"   --FILES_TRANS_COPIED\n".
	"   --FILES_TRANS_CHECKED\n".
	"   --FILES_TRANS_INSERTED\n".
	"   --FILES_DELETED\n".
	"   --FILES_ALL\n\n\n".
	"Configure directories, list of PCs, and external calls in $configFile\n\n";
    print "$message";
}

#Entry Point
foreach my $arg (@ARGV)
{
    if ("$arg" eq "-h") { &show_usage(); }
    if ("$arg" eq "--help") { &show_usage(); }
    if ("$arg" eq "--report") {
	$DO = 1;
        $DO_REPORT = 1;
    }
    if ("$arg" eq "--fullreport"){
        $DO = 1;
        $DO_REPORT = 1;
        $DO_FULL_REPORT = 1;
    }
    if ("$arg" eq "--clean"){
	$DO = 1;
	$DO_CLEAN = 1;
    }
    if ("$arg" eq "--listfiles") { $DO_FILES_EMU = 1; }
    if ("$arg" eq "--openfiles") { $DO_OPEN = 1; }
    if ("$arg" eq "--closedfiles") { $DO_CLOSED = 1; }
    if ($arg =~ /^--age=/) {
	my @parts = split(/=/, $arg);
	$DO_FILE_AGE = $parts[1];
    }
    if ($arg =~ /^--max=/) {
	my @parts = split(/=/, $arg);
	$DO_FILE_MAX = $parts[1];
    }
    if ($arg =~ /^--host=/) {
	my @parts = split(/=/, $arg);
	$DO_HOST = $parts[1];
    }
    if ($arg =~ /^--hostlist=/) {
	my @parts = split(/=/, $arg);
	$Con{"HOSTLIST"} = $parts[1];
    }
    if ($arg =~ /^--contains=/) {
	my @parts = split(/=/, $arg);
	$DO_FILE_CONTAINS = $parts[1];
    }
    if ("$arg" eq "--FILES_NO_STATUS"){
        $DO_FILES = 1;
	$DO_FILES_NO_STATUS = 1;
    }
    if ("$arg" eq "--FILES_CREATED"){
	$DO_FILES = 1;
        $DO_FILES_CREATED = 1;
    }
    if ("$arg" eq "--FILES_INJECTED"){
	$DO_FILES = 1;
        $DO_FILES_INJECTED = 1;
    }
    if ("$arg" eq "--FILES_NEW"){
	$DO_FILES = 1;
        $DO_FILES_TRANS_NEW = 1;
    }
    if ("$arg" eq "--FILES_TRANS_COPIED"){
	$DO_FILES = 1;
        $DO_FILES_TRANS_COPIED = 1;
    }
    if ("$arg" eq "--FILES_TRANS_CHECKED"){
	$DO_FILES = 1;
        $DO_FILES_TRANS_CHECKED = 1;
    }
    if ("$arg" eq "--FILES_TRANS_INSERTED"){
	$DO_FILES = 1;
        $DO_FILES_TRANS_INSERTED = 1;
    }
    if ("$arg" eq "--FILES_DELETED"){
	$DO_FILES = 1;
        $DO_FILES_DELETED = 1;
    }
    if ("$arg" eq "--FILES_ALL"){
	$DO_FILES=1;
        $DO_FILES_DELETED=1;
        $DO_FILES_CREATED=1;
        $DO_FILES_INJECTED=1;
        $DO_FILES_TRANS_NEW=1;
        $DO_FILES_TRANS_COPIED=1;
        $DO_FILES_TRANS_CHECKED=1;
        $DO_FILES_TRANS_INSERTED=1;
        $DO_FILES_DELETED=1;
    }
}

if (!$DO){
    print "Syntax error: you must select 1 action at least";
    &show_usage();
}
if (!$DO_OPEN && !$DO_CLOSED){
    print "Missing parameter: you must select 1 type of files at least (open/closed).";
    &show_usage();
}

if ($DO_REPORT){
    if ($DO_OPEN) { report($Con{"CHECK_PATH_OPEN"}); }
    if ($DO_CLOSED) {report($Con{"CHECK_PATH_CLOSED"}); }
}

if ($DO_CLEAN){
    if ($DO_FILES){
	#Confirm that clean wanted
	print "Do you really want to clean files?(y/n) ";
	my $answer = <STDIN>;
	chomp $answer;
	if ("$answer" eq "y" | "$answer" eq "yes"){
	    if ($DO_OPEN) { clean($Con{"CHECK_PATH_OPEN"}); }
	    if ($DO_CLOSED) {clean($Con{"CHECK_PATH_CLOSED"}); }
	}
    }
    else {
	print "Missing parameter: You must select 1 file status at least.";
	&show_usage();
    }
}

#Close DB
$checkHan->finish();
$entryHan->finish();
$dbh->disconnect;

