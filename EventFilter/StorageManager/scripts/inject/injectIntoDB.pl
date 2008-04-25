#!/usr/bin/perl -w
# $Id: injectIntoDB.pl,v 1.5 2008/04/25 15:40:05 loizides Exp $

use strict;
use DBI;
use Getopt::Long;

# injection subroutine
sub inject($)
{
    my $dbh = shift;

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

    my $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
        "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
        "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
        "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
        "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
        "'$type',$safety,$nevents,$filesize,$checksum)";

    if (defined $dbh) { 
        my $rows = $dbh->do($SQL) or 
            die $dbh->errstr;
        return $rows-1;
    }

    print "$SQL\n";
    return 0;
}

if (!defined $ARGV[0]) {
    die "Syntax: ./injectIntoDB.pl infile outfile";
}
my $infile="$ARGV[0]";
if (!defined $ARGV[1]) {
    die "Syntax: ./injectIntoDB.pl infile outfile";
}
my $outfile=">$ARGV[1]";

#overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
my $dbh; #my DB handle
if (!defined $ENV{'SM_DONTACCESSDB'}) { 
    my $dbi    = "DBI:Oracle:cms_rcms";
    my $reader = "CMS_STOMGR_W";
    $dbh = DBI->connect($dbi,$reader,"qwerty") or 
        die "Error: Connection to Oracle failed: $DBI::errstr\n";
}

my $line;
my $lnum = 1;
open(INDATA, $infile) or 
    die("Error: cannot open file '$infile'\n");

open(OUTDATA, $outfile) or 
    die("Error: cannot open file '$outfile'\n");

#loop over input files
while( $line = <INDATA> ){
    chomp($line);
    my @exports = split(';', $line);
    my $lexports = scalar(@exports);
    for (my $count = 0; $count < $lexports; $count++) {
        my $field = $exports[$count];
        if ($field =~ m/export (.*)=(.*)/i) {
            $ENV{$1}=$2;
        }
    }

    my $ret=inject($dbh);
    if ($ret == 0) {
        print OUTDATA "$line\n";
        my $cmd=$ENV{'SM_HOOKSCRIPT'};
        if (defined $cmd) {
            system($cmd);
        }
    }
    $lnum++;
}

close INDATA;
close OUTDATA;

# Disconnect from DB
if (defined $dbh) { 
    $dbh->disconnect or 
        warn "Warning: Disconnection from Oracle failed: $DBI::errstr\n";
}
