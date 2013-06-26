#!/usr/bin/env perl
# $Id: closeFile.pl,v 1.4 2008/05/20 08:59:06 loizides Exp $
################################################################################

use strict;
use Getopt::Long;

sub show_help {
  my $exit_status = shift@_;
  print " 
  ############################################################################## 

  Action:
  =======
  Script to update an entry to the CMS_STOMGR.TIER0_INJECTION table

  Syntax:
  ======= 
  ./closeFile.pl  --RUNNUMBER 999999  --LUMISECTION 0  --INSTANCE 0  --COUNT 0 \
                  --START_TIME  1181833642  --STOP_TIME  1181834642 \
                  --FILENAME test.00999999.0000.A.test.0.0000.dat \
                  --PATHNAME /data1/ --HOSTNAME cmsdisk1 \
                  --DATASET test --STREAM A  --STATUS closed  --TYPE streamer \
                  --SAFETY 0  --NEVENTS 999  --FILESIZE 1024  --CHECKSUM 0 
  
  ##############################################################################   
  \n";
  exit $exit_status;
}

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

my @MYARGV = @ARGV;

my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
my ($starttime,$checksum);

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

#get cmssw version from environment
my $cmsver = $ENV{'CMSSW_VERSION'};
if(length $cmsver <= 0) {
    $cmsver = "CMSSW_1_7_1";
}

# get directory for log file
my $storedir=$ENV{'SM_STORE'};
if (!defined $storedir) {
    $storedir="/store";
}

my $outdir = $storedir . "/global/log/";
if($pathname =~ m/emulator/) {
    $outdir = $storedir . "/emulator/log/";
}
if (!-d "$outdir") {
    $outdir = $pathname . "/../log/";
    if (!-d "$outdir") {
        $outdir = "/tmp/";
    }
}

my $dstr = getdatestr();
my $outfile = ">> $outdir" . $dstr . "-" . $hostname . "-" . $instance . ".log";
open LOG, $outfile;
print LOG "./closeFile.pl  --FILENAME $filename --COUNT $count --NEVENTS $nevents --FILESIZE $filesize --START_TIME $starttime --STOP_TIME $stoptime --STATUS $status --RUNNUMBER $runnumber --LUMISECTION $lumisection --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --INSTANCE $instance --SAFETY $safety --APP_VERSION $cmsver --APP_NAME CMSSW --TYPE streamer --CHECKSUM $checksum\n";
close LOG;

exit 0;
