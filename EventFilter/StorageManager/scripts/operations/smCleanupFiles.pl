#!/usr/bin/env perl
# $Id: smCleanupFiles.pl,v 1.7 2010/04/27 01:55:20 gbauer Exp $

use strict;
use warnings;
use DBI;
use Getopt::Long;
use File::Basename;
use File::Find ();


my ($help, $debug, $nothing, $now, $force, $execute, $maxfiles, $fileagemin, $skipdelete, $dbagemax);
my ($hostname, $filename, $dataset, $stream, $config);
my ($runnumber, $uptorun, $safety, $rmexitcode, $chmodexitcode );
my ($constraint_runnumber, $constraint_uptorun, $constraint_filename, $constraint_hostname, $constraint_dataset);

my ($reader, $phrase);
my ($dbi, $dbh);
my ($minRunSMI, $fileageSMI);
my %h_notfiles;



#-----------------------------------------------------------------
sub usage
{
  print "
  ##############################################################################

  Usage $0 [--help] [--debug] [--nothing]  [--now]
  Almost all the parameters are obvious. Non-obvious ones are:
   --now : suppress 'sleep' in delete based on host ID

  ##############################################################################
 \n";
  exit 0;
}

#-----------------------------------------------------------------
#subroutine for getting formatted time for SQL to_date method
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

#-----------------------------------------------------------------
sub deletefiles()
{

# Look for files in FILES_TRANS_CHECKED - implies closed and safety >= 100 in the old scheme.
# Alternate queries for different values of these? even needed?
# These files need to be in FILES_CREATED and FILES_INJECTED to
# check correct hostname and pathname. They must not be in FILES_DELETED.
my $basesql = "select PATHNAME, CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME, HOSTNAME from CMS_STOMGR.FILES_TRANS_CHECKED inner join " .
               "CMS_STOMGR.FILES_CREATED on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME inner join " .
               "CMS_STOMGR.FILES_INJECTED on CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME " .
               "where CMS_STOMGR.FILES_CREATED.CTIME+$dbagemax>sysdate  " .
               "      and not exists (select * from CMS_STOMGR.FILES_DELETED " .
                                  "where CMS_STOMGR.FILES_DELETED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME)";

# Sorting by time
my $endsql = " order by ITIME";

# Additional constraints
$constraint_runnumber = '';
$constraint_uptorun   = '';
$constraint_filename  = '';
$constraint_hostname  = '';
$constraint_dataset   = '';

if ($runnumber) { $constraint_runnumber = " and RUNNUMBER = $runnumber"; }
if ($uptorun)   { $constraint_uptorun   = " and RUNNUMBER >= $uptorun";  }
if ($filename)  { $constraint_filename  = " and CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME = '$filename'";}
if ($hostname)  { $constraint_hostname  = " and HOSTNAME = '$hostname'";}
if ($dataset)   { $constraint_dataset   = " and SETUPLABEL = '$dataset'";}

# Compose DB query
my $myquery = '';
$myquery = "$basesql $constraint_runnumber $constraint_uptorun $constraint_filename $constraint_hostname $constraint_dataset $endsql";

$debug && print "******BASE QUERY:\n   $myquery,\n";



my $insertDel = $dbh->prepare("insert into CMS_STOMGR.FILES_DELETED (FILENAME,DTIME) VALUES (?,TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))");
my $sth  = $dbh->prepare($myquery);


$sth->execute() || die "Initial DB query failed: $dbh->errstr\n";



############## Parse and process the result
my $nFiles   = 0;
my $nRMFiles = 0;
my $nRMind   = 0;



my @row;

$debug && print "MAXFILES: $maxfiles\n";

while ( $nFiles<$maxfiles &&  (@row = $sth->fetchrow_array) ) {

    $rmexitcode = 9999;    # Be over-cautious and set a non-zero default
    $debug   && print "       --------------------------------------------------------------------\n";
    $nFiles++;

    #keep track of smallest run number:
    my ($currrun) = ($row[1] =~ /[a-zA-Z0-9]+\.([0-9]+)\..*/);
    print "CurRun=$currrun  FILE: $row[1] ||\n";
    if( $minRunSMI >  $currrun ){ $minRunSMI = $currrun;}
 
    # get .ind file name
    my $file =  "$row[0]/$row[1]";
    my $fileIND;
    if ( $file =~ /^(.*)\.(?:dat|root)$/ ) {
        $fileIND = $1 . '.ind';
    }

    if ( -e $file ) {
        my $FILEAGEMIN   =  (time - (stat(_))[9])/60;
        $debug   && print "$FILEAGEMIN $fileagemin\n";

        if ($execute && $FILEAGEMIN > $fileagemin) {
            if ( unlink( $file ) == 1 ) {
                # unlink should return 1: removed 1 file
		$rmexitcode = 0;
            } else {
                print "Removal of $file failed\n";
                $rmexitcode = 9996;
           }
        } elsif (!$execute && $FILEAGEMIN > $fileagemin) {
	    #if we're not executing anything want to fake
	    print "Pretending to remove $file\n";
	    $rmexitcode = 0;
        } elsif ($FILEAGEMIN < $fileagemin) {
            print "File $file too young to die\n";
            $rmexitcode = 9995;
        } else {
            print "This should never happen. File $file has issues!\n";
            $rmexitcode = 9994;
        }
    } elsif ($force) {
        print "File $file does not exist, but force=1, so continue\n";
        $rmexitcode = 0;
    } elsif ( ! -d $row[0] ) {
        print "Path $row[0] does not exist. Are the disks mounted?\n";
        $rmexitcode = 9998;
    } else {
        print "File $file does not exist\n";
        $rmexitcode = 9997;
    }
    #$rmexitcode =0;
    # check file was really removed
    if ($rmexitcode != 0) {
	print "Could not delete file: $file (rmexitcode=$rmexitcode)\n";
    } elsif ( ! -e $file ) {
	$nRMFiles++;
	
	# insert file into deleted db
	$insertDel->bind_param(1,$row[1]);
	$insertDel->bind_param(2,gettimestamp(time));
	$execute && ($insertDel->execute() || die "DB insert into deleted files failed: $insertDel->errstr\n");
	my $delErr = $insertDel->errstr;
	if(defined($delErr)) {
	    print "Delete DB insert produced error: $delErr\n";
	} else {
	    $debug && print "File inserted into deleted DB, rm .ind\n";
	    if ( $execute && -e $fileIND) {
                my $rmIND = `rm -f $fileIND`;
                if (! -e "$fileIND" ) {$nRMind++;}
	    }
	}
    } else {
        print "Unlink returned success, but file $file is still there!\n";
    }


}


# Only print summary if STDIN is a tty, so not in cron
if( -t STDIN ) {
    print "\n=================> DONE!:\n";
    print ">>BASE QUERY WAS:\n   $myquery,\n";
    print " $nFiles Files Processed\n" .
      " $nRMFiles Files rm-ed\n" .
      " $nRMind ind Files removed\n\n\n";
}


#make sure handles are done
$sth->finish();


}


#---------------------------helper function for uncountfiles()----
sub wantedfiles {
    # Data.00129762.1825.A.storageManager.01.0000.dat
    ! /\.EcalCalibration\./
    && /(^[^.]+)\.([0-9]+)\..*\.([0-9]+)\.[0-9]+\.dat$/
    && $h_notfiles{$2}{$3}{$1}++;
}

#-----------------------------------------------------------------
sub uncountfiles(){
# routine to check disk for file unaccounted for in the DB and enter (N_UNACCOUNT) number in SM_INSTANCES table

#satadir:
    my $nsata=`df | grep -ic sata`;
    if( $nsata <1 ) { return;} 

#prepare SQL's:
#diagnostic query:
my $qinstancesql = "SELECT CMS_STOMGR.SM_INSTANCES.RUNNUMBER,SETUPLABEL,INSTANCE,HOSTNAME,N_CREATED,N_INJECTED,N_DELETED,N_UNACCOUNT " .
                           "from CMS_STOMGR.SM_INSTANCES where RUNNUMBER=? and HOSTNAME=?";

#query to zero out N_UNACCOUNT counter:
my $upallinstancesql = "UPDATE CMS_STOMGR.SM_INSTANCES SET N_UNACCOUNT=0 " .
                           "where HOSTNAME=? and ( RUNNUMBER>? or LAST_WRITE_TIME+$fileageSMI>sysdate) ";
#merge to enter info into DB:
my $mergesql = " merge into CMS_STOMGR.SM_INSTANCES using dual on (CMS_STOMGR.SM_INSTANCES.RUNNUMBER=? AND " .
               "            CMS_STOMGR.SM_INSTANCES.HOSTNAME=? AND CMS_STOMGR.SM_INSTANCES.INSTANCE=? )    " .
               " when matched then update set  N_UNACCOUNT=?-TO_NUMBER(NVL(N_INJECTED,0))+TO_NUMBER(NVL(N_DELETED,0)) " .
               " when not matched then insert (RUNNUMBER,HOSTNAME,INSTANCE,SETUPLABEL,N_UNACCOUNT) values (?,?, ?, ?, ?) ";



my $qinstance     = $dbh->prepare($qinstancesql);
my $upallinstance = $dbh->prepare($upallinstancesql);
my $merge         = $dbh->prepare($mergesql);


#check what file are ACTUALLY on disk by   desired filesystems
File::Find::find({wanted => \&wantedfiles}, </store/sata*/gcd/closed/>);


#put in and extra buffer of look at old file entries just in case for updating
$minRunSMI = $minRunSMI - 1000;

#global update to initialize counters to zero:
print "global update with minrun=$minRunSMI and age=$fileageSMI days\n";
$upallinstance->bind_param(1,$hostname);
$upallinstance->bind_param(2,$minRunSMI);
my $upallinstanceCheck = $upallinstance->execute() or die("Error: Global Zeroing of SM_INSTANCES - $dbh->errstr \n");




for my $run ( sort keys %h_notfiles ) {
    for my $instance ( sort keys %{$h_notfiles{$run}} ) {
        for my $label ( sort keys %{$h_notfiles{$run}{$instance}} ) {
	print "************ run= $run  instance=$instance  label=$label \n ";

	if($run > 129710){
	    
	    print "run= $run; instance=$instance  label=$label  files=$h_notfiles{$run}{$instance}{$label} \n";
	    

	    $qinstance->bind_param(1,$run);
	    $qinstance->bind_param(2,$hostname);
	    my $qinstanceCheck = $qinstance->execute() or die("Error: Query2 failed - $dbh->errstr \n");
	    print "CHECK: $qinstanceCheck || \n";
	    my @result = $qinstance->fetchrow_array;	
	    print "----RESULT dump:..||\n";
	    print @result;
	    print "\n---------\n";      
	    my $diff = $h_notfiles{$run}{$instance}{$label}-$result[5]+$result[6];
	    print "SELECT-OUT: $result[0], Label=$result[1], INST=$result[2], $result[3], CREA=$result[4], INJ=$result[5], DELE=$result[6], UNACC=$result[7] || diff=$diff\n";
	    
	    
##update SM_INSTANCES:
	    print "UPDATE instances for RUN $run  with disk file count=$h_notfiles{$run}{$instance}{$label} \n";
	    
	    
	    $merge->bind_param(1,$run);
	    $merge->bind_param(2,$hostname);
	    $merge->bind_param(3,$instance);
	    $merge->bind_param(4,$h_notfiles{$run}{$instance}{$label});
	    $merge->bind_param(5,$run);
	    $merge->bind_param(6,$hostname);
	    $merge->bind_param(7,$instance);
	    $merge->bind_param(8,$label);
	    $merge->bind_param(9,$h_notfiles{$run}{$instance}{$label});
	    my $mergeCheck = $merge->execute() or die("Error: Update of SM_INSTANCES for Host $hostname  Run $run - $dbh->errst \n");
	    
	    
## #check results:
	    $qinstanceCheck = $qinstance->execute() or die("Error: Query failed - $dbh->errstr \n");
	    print "CHECK: $qinstanceCheck || \n";
	    @result = $qinstance->fetchrow_array;
	    $diff = $h_notfiles{$run}{$instance}{$label}-$result[5]+$result[6];
	    print "SELECT-OUT: $result[0], Label=$result[1], INST=$result[2], $result[3], CREA=$result[4], INJ=$result[5], DELE=$result[6], UNACC=$result[7] || diff=$diff\n";


	    
	    print "-------------------\n";
## 	
	    
	    
#    }else{
#	print "Run  $run is too old to process for !Files\n";
	    
	   } 
	}
    }
}


## 
## my %touchruns=();
## foreach my $ifile (@Files) {
##     my ($run) = ($ifile =~ /[a-zA-Z0-9]+\.([0-9]+)\..*/);
##     if($minRunSMI > $run){$minRunSMI = $run;}
##     if (exists $touchruns{$run}){
## 	$touchruns{$run} =  $touchruns{$run}+1;
##     }else{
## 	$touchruns{$run} =  1;
##     }
## }
## 
## #put in and extra buffer of look at old file entries just in case for updating
## $minRunSMI = $minRunSMI - 1000;
## 
## 
## #global update to initialize counters to zero:
## print "global update with minrun=$minRunSMI and age=$fileageSMI days\n";
## $upallinstance->bind_param(1,$hostname);
## $upallinstance->bind_param(2,$minRunSMI);
## my $upallinstanceCheck = $upallinstance->execute() or die("Error: Global Zeroing of SM_INSTANCES - $dbh->errstr \n");
## 
## 
## 
## #cycle through Runs which have files on disk:
## foreach my $run (keys %touchruns) {
##     print " ********* RUN=$run, NFiles=$touchruns{$run} \n";
##     if($run > 129710){
## 
##  	$qinstance->bind_param(1,$run);
##  	$qinstance->bind_param(2,$hostname);
##  	my $qinstanceCheck = $qinstance->execute() or die("Error: Query2 failed - $dbh->errstr \n");
##  	print "CHECK: $qinstanceCheck || \n";
##  	my @result = $qinstance->fetchrow_array;	
##  	print "----RESULT dump:..||\n";
##  	print @result;
##  	print "\n---------\n";      
##         my $diff = $touchruns{$run}-$result[4]+$result[5];
##  	print "SELECT-OUT: $result[0], INST=$result[1], $result[2], CREA=$result[3], INJ=$result[4], DELE=$result[5], UNACC=$result[6] || diff=$diff\n";
##  		
## 	
## ##update SM_INSTANCES:
## 	print "UPDATE instances for RUN $run  with disk file count=$touchruns{$run} \n";
## 
##        
## 	$merge->bind_param(1,$run);
## 	$merge->bind_param(2,$hostname);
## 	$merge->bind_param(3,$touchruns{$run});
## 	$merge->bind_param(4,$run);
## 	$merge->bind_param(5,$hostname);
## 	$merge->bind_param(6,-99);
## 	$merge->bind_param(7,$touchruns{$run});
## 	my $mergeCheck = $merge->execute() or die("Error: Update of SM_INSTANCES for Host $hostname  Run $run - $dbh->errst \n");
## 	
## 	
## ## #check results:
##  	$qinstanceCheck = $qinstance->execute() or die("Error: Query failed - $dbh->errstr \n");
##  	print "CHECK: $qinstanceCheck || \n";
##  	@result = $qinstance->fetchrow_array;
##         $diff = $touchruns{$run}-$result[4]+$result[5];
##  	print "UPDATED-OUT: $result[0], INST=$result[1], $result[2], CREA=$result[3], INJ=$result[4], DELE=$result[5], UNACC=$result[6] || diff=$diff\n";
## 
##  	print "-------------------\n";
## ## 	
## 	
## 	
## #    }else{
## #	print "Run  $run is too old to process for !Files\n";
##     }
##     
## }

#keep these 'finishes' of DB at the end, so ALL worked or a rollback?
$qinstance->finish();
$upallinstance->finish();
$merge->finish();


}

#-----------------------------------------------------------------
sub deleteCopyManager()
{
    my $dir="/store/copymanager/Logs";
    
    print "search in $dir \m";
    if( -d $dir ){
	
	my $string = `df $dir | grep dev`;
	my ($favail) = ($string =~ /.*\ ([0-9]+)\%.*/);
	print "----- Initial disk usage $favail\n";
	
	#delete older than 45 days:
	my $delete =`sudo -u cmsprod find /store/copymanager/Logs/*/ -cmin +64800  -type f   -exec  rm -f \'{}\' \\\; >& /dev/null`;
	
	$string = `df $dir | grep dev`;
	($favail) = ($string =~ /.*\ ([0-9]+)\%.*/);
	print "----- 45-day disk usage $favail\n";
	
	if( $favail > 85 ){
	    
	    #delete older than 32 days:
	    $delete = `sudo -u cmsprod find /store/copymanager/Logs/*/ -cmin +46080  -type f  -exec  rm -f \'{}\' \\\; >& /dev/null`;
	    
	    $string = `df $dir | grep dev`;
	    ($favail) = ($string =~ /.*\ ([0-9]+)\%.*/);
	    print "----- 32-day disk usage $favail\n";
	    
	    
	    #brutal action: Manager files older than 15 days, and /tmp area older than 8 days
	    if( $favail > 94 ){
		$delete = `sudo -u cmsprod find /store/copymanager/Logs/*/ -cmin +21600  -type f  -exec  rm -f \'{}\' \\\; >& /dev/null`;
		$delete = `sudo -u cmsprod find /tmp/* -cmin +4320  -type f  -exec rm -f {} \; >& /dev/null`;
		
		
		#emergency action: Manager files older than 3 days, and /tmp area older than 3 days
		if( $favail > 96 ){
		    
		    $delete = `sudo -u cmsprod find /store/copymanager/Logs/*/ -cmin +4320  -type f  -exec  rm -f \'{}\' \\\; >& /dev/null`;
		    
		    $delete = `sudo -u cmsprod find /tmp/* -cmin +4320  -type f  -exec rm -f {} \; >& /dev/null`;
		    
		}
	    }
	}
	
	$string = `df $dir | grep dev`;
	($favail) = ($string =~ /.*\ ([0-9]+)\%.*/);
	print "----- FINAL disk usage $favail\n";
    }
}


###################################################################
######################### MAIN ####################################

$help       = 0;
$debug      = 0;
$nothing    = 0;
$now        = 0;
$filename   = '';
$dataset    = '';
$uptorun    = 0;
$runnumber  = 0;
$safety     = 100;
$hostname   = '';
$execute    = 1;
$maxfiles   = 1;
$fileagemin = 130;  #min
$dbagemax   = 365;  #days
$force      = 0;
$config     = "/opt/injectworker/.db.conf";


$hostname   = `hostname -s`;
chomp($hostname);

GetOptions(
           "help"          =>\$help,
           "debug"         =>\$debug,
           "nothing"       =>\$nothing,
           "now"           =>\$now,
           "force"         =>\$force,
           "config=s"      =>\$config,
           "hostname=s"    =>\$hostname,
           "run=i"         =>\$runnumber,
           "runnumber=i"   =>\$runnumber,
	   "uptorun=s"	   =>\$uptorun,
	   "filename=s"	   =>\$filename,
	   "dataset=s"	   =>\$dataset,
           "stream=s"      =>\$stream,
           "maxfiles=i"    =>\$maxfiles,
           "fileagemin=i"  =>\$fileagemin,
           "dbagemax=i"    =>\$dbagemax,
           "skipdelete"    =>\$skipdelete
	  );

$help && usage;
if ($nothing) { $execute = 0; $debug = 1; }

#setup sleep delays if any:
my  ($rack, $node);
my  $interval = 18;
my $sleeptime = 0;
if( ( ( $rack, $node ) = ( $hostname =~ /srv-c2c(0[67])-(\d+)$/i ) ) && !$now  ){
    $sleeptime = ( 2*($node-12) + 10*($rack-6) )%$interval + ($rack-6);
    print "sleep time for $hostname is $sleeptime min \n";
    sleep(60*$sleeptime);


}




$reader = "xxx";
$phrase = "xxx";
if(-e $config) {
    eval `su smpro -c "cat $config"`;
} else {
    print "Error: Can not read config file $config, exiting!\n";
    usage();
}


 $dbi    = "DBI:Oracle:cms_rcms";
 $dbh    = DBI->connect($dbi,$reader,$phrase)
    or die "Can't make DB connection: $DBI::errstr\n";



#default params for filling SM_INSTANCES table
$minRunSMI   = 999999999; #smallest run number being handled
$fileageSMI  = 3;         #in days

my $date=`date`;
print "$date ..execute DELETES...\n";
if (!$skipdelete) { deletefiles(); }
$date=`date`;
print "$date ..DONE executing DELETES...\n";




print " ..execute !Files...\n";
uncountfiles();
$date=`date`;
print "$date ..DONE executing unDELETES...\n";


$dbh->disconnect;



my $hour = `date +%H`+0;
my $min  = `date +%M`+0;
my $hourPC = ( 2*($node-12) + 10*($rack-6) )%18 + ($rack-6);
if($now || ( $hourPC == $hour && $min < $interval+10) ) { 
    deleteCopyManager(); 
}


exit 0;
