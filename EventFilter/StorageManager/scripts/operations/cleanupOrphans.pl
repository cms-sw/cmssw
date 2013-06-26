#!/usr/bin/env perl

# Script to make REPORTS and/or DELETE "global" area files abandoned on SM disk
# Script looks at disks, finds files, and if '--clean' option specified it will:
# 1) delete the file
# 2) enter file in the FILES_DELETED table if not *already* there
#    [Note: this *breaks* the continuity sequence of FILES_xxxx tables]
# 3) enter file in the FILES_ORPHANS table with:
#          COMMENT_STR = "deleted"     IF the file was     NOW inserted into the DELETED table
#                      = "redeleted"   IF the file was NOT NOW inserted into the DELETED table
#                                           because it was already(!) in the DELETED table
#                      = "faildelete"  IF the file was NOT inserted into the DELETED table
#                                         because the attempt failed for some reason!
#                      = ""            IF the file was inserted into the ORPHANS table
#                                         *before* the the above features were implemented!
#

#2010-10-10[gb]: modify to show REPACKED files too!


#Example Call:
#./cleanupOrphans.pl --max=10 --closedfiles --clean --age=30 --FILES_ALL --contains=Transfer --host=srv-C2C07-15




use warnings;
use DBI;
use Getopt::Long;
use File::Basename;

#use constant MIN_CLEAN_AGE => 7; #won't delete anything under X days
use constant MIN_CLEAN_AGE =>  5; #won't delete anything under X days




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
my $DO_FILES_TRANS_REPACKED="";
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
	", CMS_STOMGR.FILES_TRANS_REPACKED.FILENAME".
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
        "left outer join CMS_STOMGR.FILES_TRANS_REPACKED ".
	"on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_REPACKED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_TRANS_INSERTED ".
        "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_INSERTED.FILENAME " .
        "left outer join CMS_STOMGR.FILES_DELETED ".
        "on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_DELETED.FILENAME " .
        "where CMS_STOMGR.FILES_CREATED.FILENAME=?";
my $checkHan = $dbh->prepare($SQLcheck) or die("Error: DB query prepare failed - $dbh->errstr \n");


#Skeleton query to verify absence/presence in DELETED table:
my $SQLQueryDeleted = "select CMS_STOMGR.FILES_DELETED.DTIME from CMS_STOMGR.FILES_DELETED where CMS_STOMGR.FILES_DELETED.FILENAME=?";
my $QueryHanDeleted = $dbh->prepare($SQLQueryDeleted) or die("Error: DB query prepare failed - $dbh->errstr\n");



#Skeleton insert to make entry into DELETED:
my $SQLEntryDeleted = "insert into CMS_STOMGR.FILES_DELETED(FILENAME,DTIME) VALUES (?,sysdate)";
my $entryHanDeleted = $dbh->prepare($SQLEntryDeleted) or die("Error: DB query prepare failed - $dbh->errstr\n");


#Skeleton insert to make simulataneou entry into orphans table
my $SQLEntryOrphan = "insert into CMS_STOMGR.FILES_ORPHANS(FILENAME,DTIME,HOST,STATUS,COMMENT_STR) VALUES (?,sysdate,?,?,?)";
my $entryHanOrphan = $dbh->prepare($SQLEntryOrphan) or die("Error: DB query prepare failed - $dbh->errstr\n");

#Routine to get the status of a give file (based on previously existing separate script
sub getStatus($)
{
    my $filename = shift;
    my $rowsCcheck = $checkHan->execute($filename) or die("Error: Query failed - $dbh->errstr \n");
    
    my @result = $checkHan->fetchrow_array;


#####	print "$filename: \n $result[0] 1:  $result[1] 2:  $result[2] 3:  $result[3] 4:  $result[4] 5:  $result[5] 6:  $result[6] 7:  $result[7] \n"; 

    unless($result[0]) {return -1;}
    unless($result[1]) {return 0;}
    unless($result[2]) {return 1;}
    unless($result[3]) {return 10;}
    unless($result[4]) {return 20;}
#    unless($result[5]) {return 30;}
    unless($result[5]) {return 30 unless($result[7])}  #extra unless($result[7]): want to be sure to skip over gap where DELETED is still filled!
    unless($result[6]) {return 35 unless($result[7])}  #extra unless($result[7]): want to be sure to skip over gap where DELETED is still filled!
    unless($result[7]) {return 40;}
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
    my $TOTAL_FILES_TRANS_REPACKED = 0;
    my $TOTAL_FILES_TRANS_INSERTED = 0;
    my $TOTAL_FILES_DELETED = 0;
    my $TOTAL_FILES_NO_STATUS = 0;
    
    #If we want to restrict to a minimum age
    my $AGE_QUERY = "";
    if ($DO_FILE_AGE > 0) { $AGE_QUERY = "-mtime +$DO_FILE_AGE"; }

    #Keyword
    my $CONTAINS_QUERY = "";
    if ($DO_FILE_CONTAINS){ $CONTAINS_QUERY = "-name '*$DO_FILE_CONTAINS*'";}
    
    my @hosts = `cat $Con{"HOSTLIST"} | grep -i c2`;
    if ($DO_HOST){ @hosts = $DO_HOST; }
    foreach $host (@hosts ){
	my $FILES_DONE = 0;
	my $FILES_CREATED = 0;
	my $FILES_INJECTED = 0;
	my $FILES_TRANS_NEW = 0;
	my $FILES_TRANS_COPIED = 0;
	my $FILES_TRANS_CHECKED = 0;
	my $FILES_TRANS_REPACKED = 0;
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


#             my $filename1=`echo  $file | cut -d'/' -f6`;
# 		    chomp $filename1;
# 	    print "FILE: ||$filename1|| \n";
# 	     
#             my $entrycheckDeleted   = $QueryHanDeleted->execute($filename1);
#             @entryresultDELETED     = $QueryHanDeleted->fetchrow_array;
#             my $origentryErrDeleted = $QueryHanDeleted->errstr;
# 	    print "DELETED Query:  $entryresultDELETED[0]; ";
# 	    print "  $entrycheckDeleted; $origentryErrDeleted \n";
#                 if (defined($origentryErrDeleted)){ #failed to find in DB
#                     print "DELETED entry not found: error: $origentryErrDeleted \n";
#                 }
# 
#                 if (!defined($entryresultDELETED[0])){ #failed to find in DB
#                     print "DELETED entry $entryresultDELETED[0] not defined: $entryresultDELETED[0]|| \n";
#                 }
# 
# 
# 	    print " \n";



		}
		if ($status == 0) {$FILES_CREATED = $FILES_CREATED + 1;}
		elsif ($status == 1) {$FILES_INJECTED = $FILES_INJECTED + 1;}
		elsif ($status == 10) {$FILES_TRANS_NEW = $FILES_TRANS_NEW + 1;}
		elsif ($status == 20) {$FILES_TRANS_COPIED = $FILES_TRANS_COPIED + 1;}
		elsif ($status == 30) {$FILES_TRANS_CHECKED = $FILES_TRANS_CHECKED + 1;}
		elsif ($status == 35) {$FILES_TRANS_REPACKED = $FILES_TRANS_REPACKED + 1;}


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
        $TOTAL_FILES_TRANS_REPACKED = $TOTAL_FILES_TRANS_REPACKED + $FILES_TRANS_REPACKED;
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
	print "FILES_TRANS_REPACKED $FILES_TRANS_REPACKED \n";
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
    
    my $TOTAL_FILES_ELSE = 0;
    my $TOTAL_FILES_DONE = 0;
    my $TOTAL_FILES_CREATED = 0;
    my $TOTAL_FILES_INJECTED = 0;
    my $TOTAL_FILES_TRANS_NEW = 0;
    my $TOTAL_FILES_TRANS_COPIED = 0;
    my $TOTAL_FILES_TRANS_CHECKED = 0;
    my $TOTAL_FILES_TRANS_REPACKED = 0;
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
    
    my @hosts = `cat $Con{"HOSTLIST"} | grep -i c2`;
    if ($DO_HOST){ @hosts = $DO_HOST; }
    foreach $host (@hosts ){
	my $FILES_ELSE = 0;
	my $FILES_DONE = 0;
	my $FILES_CREATED = 0;
	my $FILES_INJECTED = 0;
	my $FILES_TRANS_NEW = 0;
	my $FILES_TRANS_COPIED = 0;
	my $FILES_TRANS_CHECKED = 0;
	my $FILES_TRANS_REPACKED = 0;
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
	    if ($FILES_DONE+$FILES_ELSE < $DO_FILE_MAX){
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
		elsif ($status == 35) {
		    if ($DO_FILES_TRANS_REPACKED){
			$FAILED_DELETES = $FAILED_DELETES + &delete_file($host, $file, $basename, $status);
			$FILES_TRANS_REPACKED = $FILES_TRANS_REPACKED + 1;
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
		    else{
#comment this out so "other" files (eg not in db) are not counted against the delete!			$FILES_ELSE = $FILES_ELSE + 1;
		    }
		}	      
	    } 
	}
	$COMPLETED_DELETES = $COMPLETED_DELETES - $FAILED_DELETES;

	$TOTAL_FILES_ELSE = $TOTAL_FILES_ELSE + $FILES_ELSE;
	$TOTAL_FILES_DONE = $TOTAL_FILES_DONE + $FILES_DONE;
        $TOTAL_FILES_CREATED = $TOTAL_FILES_CREATED + $FILES_CREATED;
        $TOTAL_FILES_INJECTED = $TOTAL_FILES_INJECTED + $FILES_INJECTED;
        $TOTAL_FILES_TRANS_NEW = $TOTAL_FILES_TRANS_NEW + $FILES_TRANS_NEW;
        $TOTAL_FILES_TRANS_COPIED = $TOTAL_FILES_TRANS_COPIED + $FILES_TRANS_COPIED;
        $TOTAL_FILES_TRANS_CHECKED = $TOTAL_FILES_TRANS_CHECKED + $FILES_TRANS_CHECKED;
        $TOTAL_FILES_TRANS_REPACKED = $TOTAL_FILES_TRANS_REPACKED + $FILES_TRANS_REPACKED;
        $TOTAL_FILES_TRANS_INSERTED = $TOTAL_FILES_TRANS_INSERTED + $FILES_TRANS_INSERTED;
        $TOTAL_FILES_DELETED = $TOTAL_FILES_DELETED + $FILES_DELETED;
	$TOTAL_FILES_NO_STATUS = $TOTAL_FILES_NO_STATUS + $FILES_NO_STATUS;
	
	print "FILES ON DISK: $FILES_DONE\n";
	print "\t FILES_CREATED $FILES_CREATED\n";
	print "\t FILES_INJECTED $FILES_INJECTED\n";
	print "\t FILES_TRANS_NEW $FILES_TRANS_NEW\n";
	print "\t FILES_TRANS_COPIED $FILES_TRANS_COPIED\n";
	print "\t FILES_TRANS_CHECKED $FILES_TRANS_CHECKED\n";
	print "\t FILES_TRANS_REPACKED $FILES_TRANS_REPACKED\n";
	print "\t FILES_TRANS_INSERTED $FILES_TRANS_INSERTED\n";
	print "\t FILES_DELETED $FILES_DELETED\n";
	if ($DO_FILES_NO_STATUS){
	    print "\t FILES_NO_STATUS_IN_DB $FILES_NO_STATUS\n"; 
	}
	else {
	    print "\t FILES_NO_STATUS_IN_DB(NOT DELETED) $FILES_ELSE\n";
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
	print APPLOG "            [$mday-$mon-$year $hour:$min:$sec $host]FILES_TRANS_REPACKED $FILES_TRANS_REPACKED\n";
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
	    print APPLOG "Failed: $filename\n";
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
	    print APPLOG "Failed: $filename\n";
	}
    }





    if ($rmexitcode == 0){ #successful delete, so enter in DB
	if ($targetStatus > -1){ #Can't enter in table if not already in DB
            my $orphanscomment="redeleted";

#           Check if files ALREADY in DELETED (not supposed to be, but has happened)
            my $entrycheckDeleted   = $QueryHanDeleted->execute($filename);
            @entryresultDELETED     = $QueryHanDeleted->fetchrow_array;
            if (!defined($entryresultDELETED[0])){ #file NOT in DELETED so go ahead and enter it 
####
####		print "try to enter file $filename into DELETED table \n";

#               Enter in DELETED Table:
		$entryHanDeleted->execute($filename);
		$entryErrDeleted = $entryHanDeleted->errstr;
		if (defined($entryErrDeleted)){ #failed to enter in DB
		    #Maybe exit, maybe log entry???
		    print "DELETED DB insert produced error: $entryErrDeleted \n";
	            print APPLOG "DELETED DB insert produced error: $entryErrDeleted FOR:\n";
                    $fday=`date +%c`;
                    $ftimes=`date +%s`;
	            print APPLOG "$filename  $fday  $ftimes"; 
		    $orphanscomment="faildelete";
		}
		else {
		    $orphanscomment="deleted";
####
####		    print "file was NOT in DELETED DB, re-set   orphanscomment=$orphanscomment \n";
		}
	    }else{
####
####		print "file $filename was already in DELETED table \n";
	    }
####
####		print "try to enter file $filename into ORPHANS table \n";
#           Enter in ORPHANS Table:
	    $entryHanOrphan->bind_param(1,$filename);
	    $entryHanOrphan->bind_param(2,$host);
	    $entryHanOrphan->bind_param(3,$targetStatus);
	    $entryHanOrphan->bind_param(4,$orphanscomment);
	    $entryHanOrphan->execute();
	    my $entryErr = $entryHanOrphan->errstr;
	    if (defined($entryErr)){ #failed to enter in DB
		#Maybe exit, maybe log entry???
		print "Orphan DB insert produced error: $entryErr  \n";
		print  APPLOG "Orphan DB insert produced error: $entryErr FOR: \n";
		$fday=`date +%c`;
		$ftimes=`date +%s`;
		print APPLOG "$filename || $host || $targetStatus || $orphanscomment || $fday  $ftimes"; 
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
	"   --FILES_TRANS_REPACKED\n".
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
    if ("$arg" eq "--FILES_TRANS_REPACKED"){
	$DO_FILES = 1;
        $DO_FILES_TRANS_REPACKED = 1;
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
        $DO_FILES_TRANS_REPACKED=1;
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
$entryHanOrphan->finish();
$QueryHanDeleted->finish();
$dbh->disconnect;

