#!/usr/bin/env perl
# $Id: sm_hookscript.pl,v 1.7 2009/03/05 08:54:44 jserrano Exp $
################################################################################

use strict;
use warnings;

my $filename   =  $ENV{'SM_FILENAME'};
my $count      =  $ENV{'SM_FILECOUNTER'};
my $nevents    =  $ENV{'SM_NEVENTS'};;
my $filesize   =  $ENV{'SM_FILESIZE'};
my $starttime  =  $ENV{'SM_STARTTIME'};
my $stoptime   =  $ENV{'SM_STOPTIME'};
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
my $producer    = 'StorageManager';
my $smid        = 1;

# special treatment for calibration stream
my $doca = $ENV{'SM_CALIB_NFS'};
if (defined $doca) {
    my @fields = split(m/\./,$filename);
    if($fields[3] eq "Calibration") {
        my $COPYCOMMAND = '$SMT0_BASE_DIR/sm_nfscopy.sh $SM_CALIB_NFS $SM_PATHNAME/$SM_FILENAME $SM_CALIBAREA 5';
        system($COPYCOMMAND);
    }
}

# copy first file per lumi section to look area 
my $dola = $ENV{'SM_LA_NFS'};
if (defined $dola) {
    if ($lumisection == 5 * ($instance + 1)  && $count < 1)
    {
        my $COPYCOMMAND = '$SMT0_BASE_DIR/sm_nfscopy.sh $SM_LA_NFS $SM_PATHNAME/$SM_FILENAME $SM_LOOKAREA 10';
        system($COPYCOMMAND);
    }
}

exit 0;
