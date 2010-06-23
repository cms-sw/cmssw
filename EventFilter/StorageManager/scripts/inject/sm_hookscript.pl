#!/usr/bin/env perl
# $Id: sm_hookscript.pl,v 1.20 2010/06/21 08:48:22 babar Exp $
################################################################################

# XXX This should be converted into a POE::Wheel object so it's asynchronous

use strict;
use warnings;
use Getopt::Long;

#define parameters for copy to LOOKAREA
my $lookfreq   = 20;                       #copy cycle: copy every n-th LumiSec
my $lookhosts  = 16;                       #max-number of hosts assumed
my $lookmodulo = $lookfreq * $lookhosts;

# XXX Should clean this up
my $filename    = $ENV{'SM_FILENAME'};
my $count       = $ENV{'SM_FILECOUNTER'};
my $nevents     = $ENV{'SM_NEVENTS'};
my $filesize    = $ENV{'SM_FILESIZE'};
my $starttime   = $ENV{'SM_STARTTIME'};
my $stoptime    = $ENV{'SM_STOPTIME'};
my $status      = $ENV{'SM_STATUS'};
my $runnumber   = $ENV{'SM_RUNNUMBER'};
my $lumisection = $ENV{'SM_LUMISECTION'};
my $pathname    = $ENV{'SM_PATHNAME'};
my $hostname    = $ENV{'SM_HOSTNAME'};
my $dataset     = $ENV{'SM_DATASET'};
my $stream      = $ENV{'SM_STREAM'};
my $instance    = $ENV{'SM_INSTANCE'};
my $safety      = $ENV{'SM_SAFETY'};
my $appversion  = $ENV{'SM_APPVERSION'};
my $appname     = $ENV{'SM_APPNAME'};
my $type        = $ENV{'SM_TYPE'};
my $checksum    = $ENV{'SM_CHECKSUM'};
my $setuplabel  = $ENV{'SM_SETUPLABEL'};
my $destination = $ENV{'SM_DESTINATION'};
my $index       = $ENV{'SM_INDEX'};
my $hltkey      = $ENV{'SM_HLTKEY'};
my $producer    = 'StorageManager';
my $retries     = 2;
my $copydelay   = 3;

# XXX That too
GetOptions(
    "FILENAME=s"    => \$filename,
    "PATHNAME=s"    => \$pathname,
    "HOSTNAME=s"    => \$hostname,
    "FILESIZE=i"    => \$filesize,
    "TYPE=s"        => \$type,
    "SETUPLABEL=s"  => \$setuplabel,
    "STREAM=s"      => \$stream,
    "RUNNUMBER=i"   => \$runnumber,
    "LUMISECTION=i" => \$lumisection,
    "NEVENTS=i"     => \$nevents,
    "APPNAME=s"     => \$appname,
    "APPVERSION=s"  => \$appversion,
    "STARTTIME=i"   => \$starttime,
    "STOPTIME=i"    => \$stoptime,
    "CHECKSUM=s"    => \$checksum,
    "DESTINATION=s" => \$destination,
    "INDEX=s"       => \$index,
    "HLTKEY=s"      => \$hltkey,
    "INSTANCE=i"    => \$instance,
    "FILECOUNTER=i" => \$count,
);

# XXX and that...
my $filepathname = "$pathname/$filename";

# special treatment for EcalCalibration
my $doca = $ENV{'SM_CALIB_NFS'};

if ( $stream eq "EcalCalibration" || $stream =~ '_EcalNFS$' ) {
    if ($doca) {
        my $COPYCOMMAND =
"$ENV{SMT0_BASE_DIR}/sm_nfscopy.sh $doca $filepathname $ENV{SM_CALIBAREA} 5";
        my $copyresult = 1;
        while ( $copyresult && $retries ) {
            $copyresult = system($COPYCOMMAND);
            $retries--;
            sleep($copydelay);
        }
    }
    unlink $filepathname;
    $filepathname =~ s/\.dat$/.ind/;
    unlink $filepathname;
    exit 0;
}

# copy one file per instance to look area
my $dola = $ENV{'SM_LA_NFS'};
if ($dola) {
    if (   $lumisection % $lookmodulo == ( ( $lookfreq * $instance ) + 1 )
        && $count < 1 )
    {
        my $COPYCOMMAND =
"$ENV{SMT0_BASE_DIR}/sm_nfscopy.sh $dola $filepathname $ENV{SM_LOOKAREA} 10";
        system($COPYCOMMAND);
    }
}

# delete if NoTransfer option is set
if ( $stream =~ '_NoTransfer$' ) {
    $filename = "$pathname/$filename";
    unlink $filepathname;
    $filepathname =~ s/\.dat$/.ind/;
    unlink $filepathname;
    exit 0;
}

exit 0;
