#!/usr/bin/perl -w
# $Id: notifyTier0.pl,v 1.1 2007/02/01 08:19:29 klute Exp $
################################################################################

use strict;
use Getopt::Long;
use DBI;

# log the 
open LOG, ">> /nfshome0/klute/globalRun-09-2007/log/notifyTier0.log";
print LOG scalar localtime(time),' ',join(' ',@ARGV),"\n";
close LOG;

#my ($runnumber,$lumisection,$instance,$count,$stoptime,$filename,$pathname);
#my ($hostname,$dataset,$stream,$status,$type,$safety,$nevents,$filesize);
#my ($starttime,$checksum);

#my ($CRC)        = ('0');
#my ($PRODUCER)   = ('StorageManager');
#my ($MYCHECKSUM) = ('0');
#my ($MYFILESIZE) = ('0');

#my $sender;

# get options
#GetOptions(
#           "RUNNUMBER=i"   => \$runnumber,
#           "LUMISECTION=i" => \$lumisection,
#           "INSTANCE=i"    => \$instance,
#           "COUNT=i"       => \$count,
#           "START_TIME=i"  => \$starttime,
#           "STOP_TIME=i"   => \$stoptime,
#           "FILENAME=s"    => \$filename,
#           "PATHNAME=s"    => \$pathname,
#           "HOSTNAME=s"    => \$hostname,
#           "DATASET=s"     => \$dataset,
#           "STREAM=s"      => \$stream,
#           "STATUS=s"      => \$status,
#           "TYPE=s"        => \$type,
#           "SAFETY=i"      => \$safety,
#           "NEVENTS=i"     => \$nevents,
#           "FILESIZE=i"    => \$filesize,
#           "CHECKSUM=s"    => \$checksum
#         );


# checksum
#my $FULLFILENAME = "$pathname$filename";
#($MYCHECKSUM, $MYFILESIZE, $FULLFILENAME) = split(" ",qx(cksum $FULLFILENAME));

#print LOG scalar localtime(time),"  CHECKSUM=",$MYCHECKSUM, " FILESIZE=", $MYFILESIZE, "\n";
#close LOG;

# connect to DB
#my $dbi    = "DBI:Oracle:omds";
#my $reader = "cms_sto_mgr";
#my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# do the update 
#my $SQL = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET FILESIZE=$filesize, STATUS='$status', STOP_TIME=$stoptime, NEVENTS=$nevents, CHECKSUM=$MYCHECKSUM, PATHNAME='$pathname' WHERE FILENAME = '$filename'";
#my $sth = $dbh->do($SQL);

# disconnect from DB
#$dbh->disconnect;

# forward the argument to the next script
#my $TIERZERO = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh --RUNNUMBER $runnumber --LUMISECTION $lumisection --INSTANCE $instance --COUNT $count --START_TIME $starttime --STOP_TIME $stoptime --FILENAME $filename --PATHNAME $pathname --HOSTNAME $hostname --DATASET $dataset --STREAM $stream --STATUS $status --TYPE $type --SAFETY $safety --NEVENTS $nevents --FILESIZE $filesize --CHECKSUM $MYCHECKSUM";
my $TIERZERO = "/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh --APP_NAME=StorageManager --APP_VERSION=CMSSW_1_6_0_DAQ3 @ARGV";

#print $TIERZERO;
system($TIERZERO);

exit 0;
