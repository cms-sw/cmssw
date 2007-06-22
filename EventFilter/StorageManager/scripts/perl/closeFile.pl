#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 24.
# $Id:$
################################################################################

use strict;
use Getopt::Long;
use DBI;

################################################################################

sub show_help {

  my $exit_status = shift@_;
  print " 
  ############################################################################## 

  Action:
  =======
  Script to insert an entry to the run_files table

  Syntax:
  ======= 
  ./closeFile.pl  --RUNNUMBER 999999  --LUMISECTION 0  --INSTANCE 0  --COUNT 0  
                   --START_TIME  1181833642  --STOP_TIME  1181834642 
                   --FILENAME test.00999999.0000.A.test.0.0000.dat
                   --PATHNAME /data1/ --HOSTNAME cmsdisk1
                   --DATASET test --STREAM A  --STATUS closed  --TYPE streamer               
                   --SAFETY 0  --NEVENTS 999  --FILESIZE 1024  --CHECKSUM 0 
  Example:
  ========
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

# log the 
open LOG, ">> /nfshome0/klute/globalRun-06-2007/log/closeFile.log";
print LOG scalar localtime(time),' ',join(' ',@ARGV),"\n";

my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
my ($starttime,$checksum);

my ($CRC)        = ('0');
my ($PRODUCER)   = ('StorageManager');
my ($MYCHECKSUM) = ('0');
my ($MYFILESIZE) = ('0');

my $sender;

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


# checksum
my $FULLFILENAME = "$pathname$filename";
($MYCHECKSUM, $MYFILESIZE, $FULLFILENAME) = split(" ",qx(cksum $FULLFILENAME));

print LOG scalar localtime(time),"  CHECKSUM=",$MYCHECKSUM, " FILESIZE=", $MYFILESIZE, "\n";
close LOG;

# do all this in the notification script since we have to give the checksum anyway
exit 0;

# connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");
 

# do the update 
my $SQL = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET FILESIZE=$filesize, STATUS='$status', STOP_TIME=$stoptime, NEVENTS=$nevents, CHECKSUM=$MYCHECKSUM, PATHNAME='$pathname' WHERE FILENAME = '$filename'";
my $sth = $dbh->do($SQL);

# disconnect from DB
$dbh->disconnect;

exit 0;
