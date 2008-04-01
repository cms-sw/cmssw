#!/usr/bin/env perl
# $Id: closeFile.pl,v 1.6 2008/04/01 08:50:29 loizides Exp $
################################################################################

use strict;
use Getopt::Long;
use DBI;

sub show_help {
  my $exit_status = shift@_;
  print " 
  ############################################################################## 

  Action:
  =======
  Script to insert an entry to the CMS_STOMGR.TIER0_INJECTION table

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

my $dolog = 1;
my @MYARGV = @ARGV;

my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
my ($starttime,$checksum);

my ($CRC)        = ('0');
my ($PRODUCER)   = ('StorageManager');
my ($MYCHECKSUM) = ('0');
my ($MYFILESIZE) = ('0');

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
    print LOG scalar localtime(time),": closeFile.pl ",join(' ',@MYARGV),"\n";
    close LOG;
}

################################################################################
#exit 0;

# copy first file of a run to look area 
if ( $lumisection == 1 && $count < 1 )
{
    my $COPYCOMMAND = "if test -n \"`mount | grep lookarea | grep cmsmon`\"; then test -e /lookarea && cp $pathname/$filename /lookarea && chmod a+r /lookarea/$filename; fi &"; 
   system($COPYCOMMAND);
   if ( $dolog == 1 )
   {
       my $dstr = getdatestr();
       my $ltime = localtime(time);
       my $outfile = ">> /nfshome0/smdev/logs/" . $dstr . "-" . $hostname . ".log";
       open LOG, $outfile;
       print LOG scalar localtime(time),": closeFile.pl ",$COPYCOMMAND,"\n";
       close LOG;
   }
}

# connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# do the update 
my $SQL = "UPDATE CMS_STOMGR.TIER0_INJECTION SET FILESIZE=$filesize, STATUS='$status', STOP_TIME=$stoptime, NEVENTS=$nevents, CHECKSUM=$MYCHECKSUM, PATHNAME='$pathname' WHERE FILENAME = '$filename'";

my $sth = $dbh->do($SQL);
# print $SQL;

# disconnect from DB
$dbh->disconnect;

exit 0;
