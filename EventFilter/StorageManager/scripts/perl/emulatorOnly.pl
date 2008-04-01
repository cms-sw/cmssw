#!/usr/bin/env perl
# $Id:$
################################################################################

use strict;
use Getopt::Long;
use DBI;

sub getdatestr()
{
    my @ltime = localtime(time);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;
    $year += 1900;
    $mon++;

    my $datestr=$year;
    if ($mon < 10) {
	$datestr=$datestr . "0";
    }

    $datestr=$datestr . $mon;
    if ($mday < 10) {
	$datestr=$datestr . "0";
    }
    $datestr=$datestr . $mday;
    return $datestr;
}

################################################################################

my $dolog = 1;
my @MYARGV = @ARGV;

my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
my ($starttime,$checksum);

my ($CRC)      = ('0');
my ($PRODUCER) = ('StorageManager');
my ($MYCHECKSUM) = ('0');
my ($MYFILESIZE) = ('0');

my $cmsver = $ENV{'CMSSW_VERSION'};
if(length $cmsver <= 0) {
    $cmsver = "CMSSW_1_7_1";
}

# get options
GetOptions(
           "RUNNUMBER=i"   => \$runnumber,
           "LUMISECTION=i" => \$lumisection,
           "INSTANCE=i"    => \$instance,
           "COUNT=i"       => \$count,
           "START_TIME=i"  => \$starttime,
           "STOP_TIME=i"   => \$stoptime,
           "FILENAME=s"    => \$filename,
           "PATHNAME=s"    => \$pathname,
           "HOSTNAME=s"    => \$hostname,
           "DATASET=s"     => \$dataset,
           "STREAM=s"      => \$stream,
           "STATUS=s"      => \$status,
           "TYPE=s"        => \$type,
           "SAFETY=i"      => \$safety,
           "NEVENTS=i"     => \$nevents,
           "FILESIZE=i"    => \$filesize,
           "CHECKSUM=s"    => \$checksum
          );

################################################################################

#make sure domain is stripped
my @harray = split(/\./,$hostname);
$hostname = $harray[0];

#overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

my $tmpfile = "/tmp/emuhack-" . $filename;
my $testfile1 = $tmpfile . "-1";
my $testfile2 = $tmpfile . "-2";


my $calltype = 1;
my $cmd = "touch $testfile1";
if (-e $testfile1) 
{
    $calltype = 2;
    $cmd = "rm $testfile1; touch $testfile2";
} 
elsif (-e $testfile2) 
{
    $calltype = 3;
    $cmd = "rm $testfile2";
}

system($cmd);

#print $calltype , "\n";
#exit 0;

if ( $calltype == 1 )  {
    if ( $dolog == 1 )
    {
	my $dstr = getdatestr();
	my $ltime = localtime(time);
	my $outfile = ">> /nfshome0/smdev/logs/emulator-" . $dstr . "-" . $hostname . ".log";
	open LOG, $outfile;
	print LOG "#--------------------\n";
	print LOG scalar localtime(time),": insertFile.pl ",join(' ',@MYARGV),"\n";
	close LOG;
    }
} 
elsif ( $calltype == 2 ) 
{ 
    if ( $dolog == 1 )
    {
	my $dstr = getdatestr();
	my $ltime = localtime(time);
	my $outfile = ">> /nfshome0/smdev/logs/emulator-" . $dstr . "-" . $hostname . ".log";
	open LOG, $outfile;
	print LOG scalar localtime(time),": closeFile.pl ",join(' ',@MYARGV),"\n";
	close LOG;
    }
    if ( $lumisection == 1 && $count < 1 )
    {
	my $COPYCOMMAND = "cp $pathname/$filename /lookarea && chmod a+r /lookarea/$filename &";
	system($COPYCOMMAND);
	if ( $dolog == 1 )
	{
	    my $dstr = getdatestr();
	    my $ltime = localtime(time);
	    my $outfile = ">> /nfshome0/smdev/logs/emulator-" . $dstr . "-" . $hostname . ".log";
	    open LOG, $outfile;
	    print LOG scalar localtime(time),": closeFile.pl ",$COPYCOMMAND,"\n";
	    close LOG;
	}
    }
} 
elsif ( $calltype == 3 ) 
{
    if ( $dolog == 1 )
    {
	my $dstr = getdatestr();
	my $ltime = localtime(time);
	my $outfile = ">> /nfshome0/smdev/logs/emulator-" . $dstr . "-" . $hostname . ".log";
	open LOG, $outfile;
	print LOG scalar localtime(time),": notifyTier0.pl CMSSW_VERSION=",$cmsver," ",join(' ',@MYARGV),"\n";
	my $TIERZERO = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh --APP_NAME=StorageManager --APP_VERSION=$cmsver --RUNNUMBER $runnumber --LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum";
	print LOG $TIERZERO, "\n";
	my $SQL = "DBSTRING: UPDATE CMS_STOMGR.TIER0_INJECTION SET SAFETY=1 WHERE FILENAME = '$filename'";
	print LOG $SQL, "\n";
	close LOG;
    }
}
else
{
    print "This should not happen!\n";
    exit 123; 
}

exit 0;
