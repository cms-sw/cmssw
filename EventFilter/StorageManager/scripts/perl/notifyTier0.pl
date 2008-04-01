#!/usr/bin/perl
# $Id: notifyTier0.pl,v 1.1 2007/02/01 08:19:29 klute Exp $
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

if ( $dolog == 1 )
{
    my $dstr = getdatestr();
    my $ltime = localtime(time);
    my $outfile = ">> /nfshome0/smdev/logs/" . $dstr . "-" . $hostname . ".log";
    open LOG, $outfile;
    print LOG scalar localtime(time),": notifyTier0.pl CMSSW_VERSION=",$cmsver," ",join(' ',@MYARGV),"\n";
    close LOG;
}

################################################################################
#exit 0;

# forward the argument to the next script
my $TIERZERO = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh --APP_NAME=StorageManager --APP_VERSION=$cmsver --RUNNUMBER $runnumber --LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $checksum";

system($TIERZERO);
#print $TIERZERO;

# connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# do the update 
my $SQL = "UPDATE CMS_STOMGR.TIER0_INJECTION SET SAFETY=1 WHERE FILENAME = '$filename'";

my $sth = $dbh->do($SQL);
#print $SQL;

# disconnect from DB
$dbh->disconnect;

exit 0;
