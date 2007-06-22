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
  ./insertFile.pl  --RUNNUMBER 999999  --LUMISECTION 0  --INSTANCE 0  --COUNT 0  
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

open LOG, ">> /nfshome0/klute/globalRun-06-2007/log/insert.log";
print LOG scalar localtime(time),' ',join(' ',@ARGV),"\n";
close LOG;

my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
my ($starttime,$checksum);

my ($CRC)      = ('0');
my ($PRODUCER) = ('StorageManager');

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


# connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");
 
# do the update
my $SQL = "INSERT INTO CMS_STO_MGR_ADMIN.TIER0_INJECTION (RUNNUMBER,LUMISECTION,INSTANCE,PRODUCER,PATHNAME,FILENAME,HOSTNAME,STREAM,DATASET,STATUS,NEVENTS,FILESIZE,START_TIME,STOP_TIME,CHECKSUM,CRC,SAFETY,COUNT,TYPE) VALUES ($runnumber,$lumisection,$instance,'$PRODUCER','$pathname','$filename','$hostname','$stream','$dataset','$status',$nevents,$filesize,'$starttime','$stoptime',$checksum,$CRC,$safety,$count,'$type')";
my $sth = $dbh->do($SQL);

# print $SQL;
# Disconnect from DB
$dbh->disconnect;

exit 0;
