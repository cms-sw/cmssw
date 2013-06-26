#!/usr/bin/env perl

use strict;
use warnings;
use DBI;
use Getopt::Long;
use File::Basename;

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

my $startRun = 0;
my $stopRun = 101000;
#my $reader = "CMS_STOMGRTEST";
#my $phrase = "zxcvbnm,";
my $reader = "CMS_STOMGR";
my $phrase = "cmsonr2008";

my $dbi = "DBI:Oracle:cms_rcms";
my $dbh = DBI->connect($dbi,$reader,$phrase);

my $checkSMsql = "select RUNNUMBER from SM_SUMMARY where RUNNUMBER=?";
my $check = $dbh->prepare($checkSMsql);

my $loadsql = "select * from CMS_STOMGR.FILES_INFO where RUNNUMBER=?";
my $sth = $dbh->prepare($loadsql);

my $runnumber;
for ($runnumber = $startRun; $runnumber <= $stopRun; $runnumber++){

    $check->bind_param(1,$runnumber);
    $check->execute();
    
    if ($check->fetchrow_array){
	print "\nRunnumber $runnumber already exists";
	next;
    }
    
    $sth->bind_param(1, $runnumber);
    $sth->execute();

    my @row;
    
    if (!(@row = $sth->fetchrow_array)){
	print "\nNo existing data for runnumber $runnumber";
	next;
    }
    
    my ($setupLabel, $appVersion, $HLTKey, $s_lumi, $s_size, $s_size2D, $s_size2T0, $s_nEvents);
    my ($s_created, $s_injected, $s_new, $s_copied, $s_checked, $s_inserted, $s_repacked, $s_deleted);
    my ($start_write, $stop_write, $start_trans, $stop_trans, $start_repack, $stop_repack);

    $s_lumi = 0;
    $s_size = 0;
    $s_size2D = 0;
    $s_size2T0 = 0;
    $s_nEvents = 0;
    $s_created = 0;
    $s_injected = 0;
    $s_new = 0;
    $s_copied = 0;
    $s_checked = 0;
    $s_inserted = 0;
    $s_repacked = 0;
    $s_deleted = 0;

    do {
	my $state = $row[1];
	if (!$setupLabel) {
	    $setupLabel = $row[3];
	}

	if (!$appVersion){
	    my $app = $dbh->prepare("SELECT APP_VERSION from CMS_STOMGR.FILES_CREATED where FILENAME='$row[0]'");
	    $app->execute();
	    my @appResult;
	    if (@appResult = $app->fetchrow_array){
		$appVersion = $appResult[0];
	    }
	}

	if (!$HLTKey) {
	    my $hlt = $dbh->prepare("SELECT COMMENT_STR from CMS_STOMGR.FILES_INJECTED where FILENAME='$row[0]'");
	    $hlt->execute();
	    my @hltResult;
	    if (@hltResult = $hlt->fetchrow_array){
		$HLTKey = $hltResult[0];
	    }
	}

	$s_lumi = $s_lumi + $row[6];
	if ($state > 0){
	    $s_size = $s_size + $row[8];
	}
	if ($state > 0){
	    $s_size2D = $s_size2D + $row[8];
	}
	if ($state > 10){
	    $s_size2T0 = $s_size2T0 + $row[8];
	}
	if ($state > 0)
	{
	    $s_nEvents = $s_nEvents + $row[7];
	}
	if ($row[9]) {
	    $s_created++;
	}
	if ($row[10]) {
	    $s_injected++;
	}
	if ($row[15]) {
	    $s_new++;
	}
	if ($row[13]) {
	    $s_copied++;
	}
	if ($row[12]) {
	    $s_checked++;
	}
	if ($row[14]) {
	    $s_inserted++;
	}
	if ($row[16]) {
	    $s_repacked++;
	}
	if ($row[11]) {
	    $s_deleted++;
	}
    } while (@row = $sth->fetchrow_array);

    my $insertsql = "insert into SM_SUMMARY (RUNNUMBER,SETUPLABEL,APP_VERSION,S_LUMISECTION,S_FILESIZE,S_FILESIZE2D,S_FILESIZE2T0,S_NEVENTS,S_CREATED,S_INJECTED,S_NEW,S_COPIED,S_CHECKED,S_INSERTED,S_REPACKED,S_DELETED,M_INSTANCE,START_WRITE_TIME,STOP_WRITE_TIME,START_TRANS_TIME,STOP_TRANS_TIME,START_REPACK_TIME,STOP_REPACK_TIME,HLTKEY,LAST_UPDATE_TIME) VALUES ($runnumber,'$setupLabel','$appVersion',$s_lumi,$s_size,$s_size2D,$s_size2T0,$s_nEvents,$s_created,$s_injected,$s_new,$s_copied,$s_checked,$s_inserted,$s_repacked,$s_deleted,".
	"(select MAX(INSTANCE) from CMS_STOMGR.FILES_CREATED where runnumber=$runnumber),".
	"(select MIN(CREATED_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber and state>0),".
	"(select MAX(INJECTED_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber and state>0),".
	"(select MIN(NEW_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber and state>10),".
	"(select MAX(COPIED_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber and state>10),".
	"(select MIN(CHECKED_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber),".
	"(select MAX(REPACKED_TIME) from CMS_STOMGR.FILES_INFO where runnumber=$runnumber),".
	"'$HLTKey',sysdate)";

    my $insertRun = $dbh->prepare($insertsql);
    $insertRun->execute(); 
    

    print "\nEntry created for runnumber $runnumber";

    sleep(1);
}
$sth->finish();
$dbh->disconnect;
