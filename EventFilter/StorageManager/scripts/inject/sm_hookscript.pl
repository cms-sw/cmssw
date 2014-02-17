#!/usr/bin/env perl
# $Id: sm_hookscript.pl,v 1.26 2012/04/09 12:03:52 babar Exp $
################################################################################

# XXX This should be converted into a POE::Wheel object so it's asynchronous

use strict;
use warnings;
use Getopt::Long;

#define parameters for copy to LOOKAREA
my $lookfreq   = 20;                       #copy cycle: copy every n-th LumiSec
my $lookhosts  = 16;                       #max-number of hosts assumed
my $lookmodulo = $lookfreq * $lookhosts;

# Global defaults, overridden depending on streams
my $copycommand = "$ENV{SMT0_BASE_DIR}/sm_nfscopy.sh";
my $nfsserver   = '';
my $target      = '';
my $parallel  = 10;                        # Allow 10 instances of sm_nfscopy.sh
my $retries   = 1;                         # Default: do not retry
my $copydelay = 3;

my (
    $appname,  $appversion, $runnumber,   $lumisection, $filename,
    $pathname, $hostname,   $destination, $setuplabel,  $stream,
    $type,     $nevents,    $filesize,    $checksum,    $instance,
    $hltkey,   $starttime,  $stoptime,    $count
);

# XXX Could clean this up as most of those are not used, but POE::Wheel will do
GetOptions(
    "APPNAME=s"     => \$appname,
    "APPVERSION=s"  => \$appversion,
    "RUNNUMBER=i"   => \$runnumber,
    "LUMISECTION=i" => \$lumisection,
    "FILENAME=s"    => \$filename,
    "PATHNAME=s"    => \$pathname,
    "HOSTNAME=s"    => \$hostname,
    "DESTINATION=s" => \$destination,
    "SETUPLABEL=s"  => \$setuplabel,
    "STREAM=s"      => \$stream,
    "TYPE=s"        => \$type,
    "NEVENTS=i"     => \$nevents,
    "FILESIZE=i"    => \$filesize,
    "CHECKSUM=s"    => \$checksum,
    "INSTANCE=i"    => \$instance,
    "HLTKEY=s"      => \$hltkey,
    "STARTTIME=i"   => \$starttime,
    "STOPTIME=i"    => \$stoptime,
    "FILECOUNTER=i" => \$count,
);

my $filepathname = "$pathname/$filename";
my $delete = $stream =~ '_NoTransfer$';

# special treatment for EcalCalibration
if ( $stream eq "EcalCalibration" || $stream =~ '_EcalNFS$' ) {
    $nfsserver = $ENV{'SM_CALIB_NFS'};
    $target =
      $ENV{SM_CALIBAREA} . '/'
      . ( $hostname eq 'srv-C2D05-02' ? 'minidaq' : 'global' );
    $parallel = 5;
    $retries  = 2;    # Retry once
    $delete   = 1;
}

# special treatment for Error, so HLT DOC can read them
elsif ( $stream eq "Error" ) {
    $nfsserver = $ENV{SM_LOOKAREA};
    $target    = $ENV{SM_LOOKAREA} . '/Error';
    $parallel  = 5;
    $retries   = 2;                              # Retry once
}

# copy one file per instance to look area
elsif ( $nfsserver = $ENV{'SM_LA_NFS'} ) {
    if (   $lumisection % $lookmodulo == ( ( $lookfreq * $instance ) + 1 )
        && $count < 1 )
    {
        $target   = $ENV{SM_LOOKAREA};
        $parallel = 10;
    }
}

if ( $nfsserver && $target ) {
    while (
        system( $copycommand, $nfsserver, $filepathname, $target, $parallel )
        && $retries-- )
    {
        sleep($copydelay) if $retries;
    }
}

# delete if NoTransfer option is set or it's EcalCalibration
if ( $delete ) {
    unlink $filepathname;
    $filepathname =~ s/\.dat$/.ind/;
    unlink $filepathname;
}

exit 0;
