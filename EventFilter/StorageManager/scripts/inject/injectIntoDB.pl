#!/usr/bin/perl -w
# $Id:$

use strict;
#use DBI;
use Getopt::Long;

# injection subroutine
sub inject()
{
    my $filename=$ENV{'SM_FILENAME'};
    my $count=$ENV{'SM_FILECOUNTER'};
    my $nevents=$ENV{'SM_NEVENTS'};;
    my $filesize=$ENV{'SM_FILESIZE'};
    my $starttime=$ENV{'SM_STARTTIME'};
    my $stoptime=$ENV{'SM_STOPTIME'};
    my $status=$ENV{'SM_STATUS'};
    my $runnumber=$ENV{'SM_RUNNUMBER'};
    my $lumisection=$ENV{'SM_LUMISECTION'};
    my $pathname=$ENV{'SM_PATHNAME'};
    my $hostname=$ENV{'SM_HOSTNAME'};
    my $dataset=$ENV{'SM_DATASET'};
    my $stream=$ENV{'SM_STREAM'};
    my $instance=$ENV{'SM_INSTANCE'};
    my $safety=$ENV{'SM_SAFETY'};
    my $appversion=$ENV{'SM_APPVERSION'};
    my $appname=$ENV{'SM_APPNAME'};
    my $type=$ENV{'SM_TYPE'};
    my $checksum=$ENV{'SM_CHECKSUM'};
    my $producer='StorageManager';

    my $SQL = "INSERT INTO CMS_STOMGR.TIER0_INJECTION (" .
        "RUNNUMBER,LUMISECTION,INSTANCE,COUNT,START_TIME,STOP_TIME,FILENAME,PATHNAME," .
        "HOSTNAME,DATASET,PRODUCER,STREAM,STATUS,TYPE,SAFETY,NEVENTS,FILESIZE,CHECKSUM) " .
        "VALUES ($runnumber,$lumisection,$instance,$count,'$starttime','$stoptime'," .
        "'$filename','$pathname','$hostname','$dataset','$producer','$stream','$status'," .
        "$type,$safety,$nevents,$filesize,$checksum)";

    print "$SQL\n";
    return 0;
}

my $infile="$ARGV[0]";
my $outfile=">$ARGV[1]";

#overwrite TNS to be sure it points to new DB
$ENV{'TNS_ADMIN'} = '/etc/tnsnames.ora';

# connect to DB
#my $dbi    = "DBI:Oracle:cms_rcms";
#my $reader = "CMS_STOMGR_W";
#my $dbh    = DBI->connect($dbi,$reader,"qwerty") or 
#    die "Error: Connection to Oracle failed: $DBI::errstr\n";

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

    my $ret=inject();
    if ($ret == 0) {
        print OUTDATA "$line\n";
    }
    $lnum++;
}

close INDATA;
close OUTDATA;

# Disconnect from DB
#$dbh->disconnect or 
#    warn "Warning: Disconnection from Oracle failed: $DBI::errstr\n";



# copy first file of a run to look area 
#if ( $lumisection == 1 && $count < 1 )
#{
#    my $COPYCOMMAND = "if test -n \"`mount | grep lookarea | grep cmsmon`\"; then test -e /lookarea && cp $pathname/$filename /lookarea && chmod a+r /lookarea/$filename; fi &"; 
#   system($COPYCOMMAND);
#   if ( $dolog == 1 )
#   {
#       my $dstr = getdatestr();
#       my $ltime = localtime(time);
#       my $outfile = ">> /nfshome0/smdev/logs/" . $dstr . "-" . $hostname . ".log";
#       open LOG, $outfile;
#       print LOG scalar localtime(time),": closeFile.pl ",$COPYCOMMAND,"\n";
#       close LOG;
#   }
#}
