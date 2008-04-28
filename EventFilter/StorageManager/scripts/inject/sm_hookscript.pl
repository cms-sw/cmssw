#!/usr/bin/perl -w
# $Id: closeFile.pl,v 1.7 2008/04/01 10:59:53 loizides Exp $
################################################################################

use strict;

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

# copy first file per lumi section to look area 
my $dola = $ENV{'SM_LA_NFS'}
if (defined $dola) {
    if ($lumisection == 1 && $count < 1)
    {
        my $COPYCOMMAND = "if test -n \"`mount | grep $SM_LA_NFS`\"; then cp $pathname/$filename $SM_LOOKAREA && chmod a+r $SM_LOOKAREA$filename; fi &"; 
        system($COPYCOMMAND);
    }
}

exit 0;
