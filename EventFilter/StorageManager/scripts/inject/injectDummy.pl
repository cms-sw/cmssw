#!/usr/bin/perl -w
# $Id: injectDummy.pl,v 1.1 2008/04/29 21:43:22 loizides Exp $

use strict;
use Getopt::Long;

# fake injection subroutine
sub inject()
{
    my $filename    = $ENV{'SM_FILENAME'};
    my $count       = $ENV{'SM_FILECOUNTER'};
    my $nevents     = $ENV{'SM_NEVENTS'};;
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
    my $producer    = 'StorageManager';

    my $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
        "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
        "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
        "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
        "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
        "'$type',$safety,$nevents,$filesize,$checksum)";

    print "$SQL\n";
    return 0;
}

# main starts here
if (!defined $ARGV[0]) {
    die "Syntax: ./injectIntoDB.pl infile outfile";
}
my $infile="$ARGV[0]";
if (!defined $ARGV[1]) {
    die "Syntax: ./injectIntoDB.pl infile outfile";
}
my $outfile=">$ARGV[1]";

my $line;
my $lnum = 1;
open(INDATA, $infile) or 
    die("Error: cannot open file '$infile'\n");

open(OUTDATA, $outfile) or 
    die("Error: cannot open file '$outfile'\n");

#loop over input files
while( $line = <INDATA> ){
    chomp($line);
    if ($line =~ m/export/i) {
        my @exports = split(';', $line);
        my $lexports = scalar(@exports);
        for (my $count = 0; $count < $lexports; $count++) {
            my $field = $exports[$count];
            if ($field =~ m/export (.*)=(.*)/i) {
                $ENV{$1}=$2;
            }
        }
    } elsif ($line =~ m/\-\-/i) {
        my @exports = split(' ', $line);
        my $lexports = scalar(@exports);
        for (my $count = 0; $count < $lexports/2; $count++) {
            my $field = "SM_$exports[2*$count]=$exports[2*$count+1]";
            if ($field =~ m/\-\-(.*)=(.*)/i) {
                my $fname = "SM_$1";
                if    ($1 eq "COUNT")      { $fname = "SM_FILECOUNTER";}
                elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
                elsif ($1 eq "STOP_TIME")  { $fname = "SM_STOPTIME";}
                elsif ($1 eq "START_TIME") { $fname = "SM_STARTTIME";}
                $ENV{$fname}=$2;
            }
        } 
    }
    my $ret=inject();
    if ($ret == 0) {
        print OUTDATA "$line\n";
    }
    $lnum++;
}

close INDATA;
close OUTDATA;
