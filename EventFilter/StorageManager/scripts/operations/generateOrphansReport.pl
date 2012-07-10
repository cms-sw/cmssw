#!/usr/bin/env perl

#Sample Call:
#./generateOrphansReport.pl -user=CMS_STOMGR_W -pass=xxxxx -stopdeletedate=1-AUG-09 -contains='Transfer' -listfiles

use strict;
use warnings;
use DBI;
use Getopt::Long;
use File::Basename;

my $startRun = "";
my $stopRun = "";
my $startCreateDate = "";
my $stopCreateDate = "";
my $startDeleteDate = "";
my $stopDeleteDate = "";
my $host = "";
my $nothost = "";
my $contains = "";
my $notcontains = "";
my $listfiles = "";
my $hostlist = "";

#Set up Database Stuff
my $dbconfig = "";
my $reader = "XXX";
my $phrase = "xxx";

GetOptions(
	   "startrun=i"         => \$startRun,
	   "stoprun=i"          => \$stopRun,
	   "startcreatedate=s"  => \$startCreateDate,
	   "stopcreatedate=s"   => \$stopCreateDate,
	   "startdeletedate=s"  => \$startDeleteDate,
	   "stopdeletedate=s"   => \$stopDeleteDate,
	   "host=s"             => \$host,
	   "nothost=s"          => \$nothost,
	   "hostlist=s"         => \$hostlist,
	   "user=s"             => \$reader,
	   "pass=s"             => \$phrase,
	   "dbconfig=s"           => \$dbconfig,
	   "contains=s"         => \$contains,
	   "notcontains=s"      => \$notcontains,
	   "listfiles"          => \$listfiles
	   );

if ($dbconfig){
    if (-e $dbconfig){
	eval `sudo -u smpro cat $dbconfig`;
    }
    else{
	die("Error: Unable to access database user information");
    }
}

my $dbi = "DBI:Oracle:cms_rcms";
my $dbh = DBI->connect($dbi,$reader,$phrase);

my $conditions = "";
if ($startRun){ $conditions = "WHERE CMS_STOMGR.FILES_CREATED.RUNNUMBER >= $startRun";}
if ($stopRun){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_CREATED.RUNNUMBER <= $stopRun";
}
if ($startCreateDate){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_CREATED.CTIME >= '$startCreateDate'";
}
if ($stopCreateDate){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_CREATED.CTIME <= '$stopCreateDate'";
}
if ($startDeleteDate){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_ORPHANS.DTIME >= '$startDeleteDate'";
}
if ($stopDeleteDate){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_ORPHANS.DTIME <= '$stopDeleteDate'";
}
if ($host){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_ORPHANS.HOST = '$host'";
}
if ($nothost){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_ORPHANS.HOST != '$nothost'";
}
if ($hostlist){
    if (! -e $hostlist){
	die "ERROR: Hostlist not found";
    }
    my @hosts = `cat $hostlist`;
    if ($hosts[0]){ #At least 1 host in list
	my $list = "(";
	foreach $host (@hosts){
	    chomp $host;
	    $list = "$list'$host',";
	}
	substr($list, length( $list ) - 1, 1) = ')';
	if ($conditions){$conditions = "$conditions AND";} 
	else {$conditions = "WHERE"};
	$conditions = "$conditions CMS_STOMGR.FILES_ORPHANS.HOST IN $list";
    }
}
    
if ($contains){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_CREATED.FILENAME LIKE '%$contains%'";
}
if ($notcontains){
    if ($conditions){$conditions = "$conditions AND";} 
    else {$conditions = "WHERE"};
    $conditions = "$conditions CMS_STOMGR.FILES_CREATED.FILENAME NOT LIKE '%$notcontains%'";
}


my $query = "select CMS_STOMGR.FILES_CREATED.FILENAME, CMS_STOMGR.FILES_CREATED.RUNNUMBER, CMS_STOMGR.FILES_CREATED.CTIME, CMS_STOMGR.FILES_INJECTED.FILESIZE, CMS_STOMGR.FILES_ORPHANS.DTIME, CMS_STOMGR.FILES_ORPHANS.HOST, CMS_STOMGR.FILES_ORPHANS.STATUS FROM CMS_STOMGR.FILES_ORPHANS left outer join CMS_STOMGR.FILES_CREATED on CMS_STOMGR.FILES_ORPHANS.FILENAME=CMS_STOMGR.FILES_CREATED.FILENAME left outer join CMS_STOMGR.FILES_INJECTED on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME $conditions";

my $queryHan = $dbh->prepare($query);
$queryHan->execute();

my $FILES_DONE = 0;
my $FILES_CREATED = 0;
my $FILES_INJECTED = 0;
my $FILES_TRANS_NEW = 0;
my $FILES_TRANS_COPIED = 0;
my $FILES_TRANS_CHECKED = 0;
my $FILES_TRANS_INSERTED = 0;
my $FILES_DELETED = 0;
my $TOTAL_FILESIZE = 0;

my @row;
while (@row = $queryHan->fetchrow_array){
    my $run = $row[1];
    my $file = $row[0];
    my $ctime = $row[2];
    my @cparts = split(/ /,$ctime);
    my $cdate = $cparts[0]; 
    my $size = $row[3] / (1024 * 1024);
    my $dtime = $row[4];
    my @dparts = split(/ /,$dtime);
    my $ddate = $dparts[0];
    my $machine = $row[5];
    my $status = $row[6];
    if (!$size) {$size = 0;}
    my $roundSize = int($size);
    if ($listfiles) {print "$cdate  $ddate  $run  $machine  $status  $roundSize MB  $file\n";}
    
    if ($status == 0) {$FILES_CREATED = $FILES_CREATED + 1;}
    elsif ($status == 1) {$FILES_INJECTED = $FILES_INJECTED + 1;}
    elsif ($status == 10) {$FILES_TRANS_NEW = $FILES_TRANS_NEW + 1;}
    elsif ($status == 20) {$FILES_TRANS_COPIED = $FILES_TRANS_COPIED + 1;}
    elsif ($status == 30) {$FILES_TRANS_CHECKED = $FILES_TRANS_CHECKED + 1;}
    elsif ($status == 40) {$FILES_TRANS_INSERTED = $FILES_TRANS_INSERTED + 1;}
    elsif ($status == 99) {$FILES_DELETED = $FILES_DELETED + 1;}
    else {}
    
    $FILES_DONE = $FILES_DONE + 1;
    $TOTAL_FILESIZE = $TOTAL_FILESIZE + $size;
}

print "File Categories: \n";
print "\tFILES_CREATED $FILES_CREATED \n";
print "\tFILES_INJECTED $FILES_INJECTED \n";
print "\tFILES_TRANS_NEW $FILES_TRANS_NEW \n";
print "\tFILES_TRANS_COPIED $FILES_TRANS_COPIED \n";
print "\tFILES_TRANS_CHECKED $FILES_TRANS_CHECKED \n";
print "\tFILES_TRANS_INSERTED $FILES_TRANS_INSERTED \n";
print "\tFILES_DELETED $FILES_DELETED \n";
print "Total Files $FILES_DONE \n";
my $roundTotalFilesize = int($TOTAL_FILESIZE);
print "Total Size $roundTotalFilesize MB\n";

$queryHan->finish();
$dbh->disconnect;
