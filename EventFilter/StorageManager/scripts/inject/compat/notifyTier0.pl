#!/usr/bin/perl
# $Id: notifyTier0.pl,v 1.5 2008/05/20 08:59:06 loizides Exp $
################################################################################

use strict;
use Getopt::Long;

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
    if ($hour < 10) {
	$datestr=$datestr . "0";
    }
    $datestr=$datestr . $hour;
    $datestr=$datestr . "-";
    my $mc=eval "($min-$min%5)/5";
    if ($mc < 10) {
	$datestr=$datestr . "0";
    }
    $datestr=$datestr . $mc;

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

my $outdir = $storedir . "/global/mbox/";
if($pathname =~ m/emulator/) {
    $outdir = $storedir . "/emulator/mbox/";
}
if (!-d "$outdir") {
    $outdir = $pathname . "/../mbox/";
    if (!-d "$outdir") {
        $outdir = "/tmp/";
    }
}

my $dstr = getdatestr();
my $outfile = ">> $outdir" . $hostname . "-" . $instance . "-" . $dstr . ".work_in_progress";
open LOG, $outfile;
print LOG "export SM_FILENAME=$filename; export SM_FILECOUNTER=$count; export SM_NEVENTS=$nevents; export SM_FILESIZE=$filesize; export SM_STARTTIME=$starttime; export SM_STOPTIME=$stoptime; export SM_STATUS=$status; export SM_RUNNUMBER=$runnumber; export SM_LUMISECTION=$lumisection; export SM_PATHNAME=$pathname; export SM_HOSTNAME=$hostname; export SM_DATASET=$dataset; export SM_STREAM=$stream; export SM_INSTANCE=$instance; export SM_SAFETY=$safety; export SM_APPVERSION=$cmsver; export SM_APPNAME=CMSSW; export SM_TYPE=streamer; export SM_CHECKSUM=0\n";

close LOG;

exit 0;
