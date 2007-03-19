#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 24.
# $Id:$
################################################################################

use strict;
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
  insertFile.pl <RUNNUMBER> <LUMISECTION> <INSTANCE> <PRODUCER> <PATHNAME> <FILENAME>\
    <HOSTNAME> <STREAM> <DATASET> <STATUS> <NEVENTS> <FILESIZE> <START_TIME>\
    <STOP_TIME> <CHECKSUM> <CRC> <SAFETY> <COUNT> <TYPE>
  - h          to get this help
 
  Example:
  ========
  ./insertFile.pl '43000' '0' '0' 'StorageManager' '/data/' \ 
         'test.00000000.A.StorageManager' 'cmsdisk0' \
         'A' '0' 'open' '0' '0' '1171550007' \
         '1171550007' '0' '0' '0' '0' 'test'
  inserts dummy entry in run_files table
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($RUNNUMBER)   = ('0');  
my ($LUMISECTION) = ('0');  
my ($INSTANCE)    = ('0');
#
my ($PRODUCER)    = ('StorageManager');  
my ($PATHNAME)    = ('/data/');  
my ($FILENAME)    = ('test.00000000.A.StorageManager');
my ($HOSTNAME)    = ('cmsdisk0');  
my ($STREAM)      = ('A');  
my ($DATASET)     = ('test');  
my ($STATUS)      = ('open');  
my ($NEVENTS)     = ('0');  
my ($FILESIZE)    = ('0');  
my ($START_TIME)  = ('1171550007');  
my ($STOP_TIME)   = ('1171550007');
my ($CHECKSUM)    = ('0');
my ($CRC)         = ('0');
my ($SAFETY)      = ('0');
my ($COUNT)       = ('0');  
my ($TYPE)        = ('streamer');  


if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if ($#ARGV ==  18)      { $RUNNUMBER   = "$ARGV[0]";
			  $LUMISECTION = "$ARGV[1]";
			  $INSTANCE    = "$ARGV[2]";  
			  $PRODUCER    = "$ARGV[3]";  
			  $PATHNAME    = "$ARGV[4]";    
			  $FILENAME    = "$ARGV[5]";    
			  $HOSTNAME    = "$ARGV[6]";    
			  $STREAM      = "$ARGV[7]";      
			  $DATASET     = "$ARGV[8]";     
			  $STATUS      = "$ARGV[9]";      
			  $NEVENTS     = "$ARGV[10]";     
			  $FILESIZE    = "$ARGV[11]";    
			  $START_TIME  = "$ARGV[12]";  
			  $STOP_TIME   = "$ARGV[13]";   
			  $CHECKSUM    = "$ARGV[14]";    
			  $CRC         = "$ARGV[15]"; 
		          $SAFETY      = "$ARGV[16]";
		          $COUNT       = "$ARGV[17]";
		          $TYPE        = "$ARGV[18]"
			  }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Do the update
my $SQLUPDATE = "INSERT INTO CMS_STO_MGR_ADMIN.TIER0_INJECTION (RUNNUMBER,LUMISECTION,INSTANCE,PRODUCER,PATHNAME,FILENAME,HOSTNAME,STREAM,DATASET,STATUS,NEVENTS,FILESIZE,START_TIME,STOP_TIME,CHECKSUM,CRC,SAFETY,COUNT,TYPE) VALUES ($RUNNUMBER,$LUMISECTION,$INSTANCE,'$PRODUCER','$PATHNAME','$FILENAME','$HOSTNAME','$STREAM','$DATASET','$STATUS',$NEVENTS,$FILESIZE,'$START_TIME','$STOP_TIME',$CHECKSUM,$CRC,$SAFETY,$COUNT,'$TYPE')";

my $sth = $dbh->do($SQLUPDATE);
#print "$SQLUPDATE \n";

# Disconnect from DB
$dbh->disconnect;
exit 0;
