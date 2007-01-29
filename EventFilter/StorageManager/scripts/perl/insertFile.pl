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
  insertFile <RUNNUMBER><LUMISECTION><INSTANCE><PRODUCER><PATHNAME><FILENAME>\
    <HOSTNAME><STREAM><DATASET><STATUS><NEVENTS><FILESIZE><START_TIME>\
    <STOP_TIME><CHECKSUM><CRC>
  - h          to get this help
 
  Example:
  ========
  ./insertFile '0' '0' '0' 'StorageManager' '/data/' \ 
         'test.00000000.A.StorageManager' 'cmsdisk0' \
         'A' '0' 'open' '0' '0' '24-DEC-2006 12.59.59.00' \
         '25-DEC-2006 01.00.00.00' '0' '0'
  inserts dummy entry in run_files table
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($RUNNUMBER)   = ('0');  
my ($LUMISECTION) = ('0');  
my ($INSTANCE)    = ('0');
#my ($COUNT)      =  ('0');  
my ($PRODUCER)    = ('StorageManager');  
my ($PATHNAME)    = ('/data/');  
my ($FILENAME)    = ('test.00000000.A.StorageManager');
my ($HOSTNAME)    = ('cmsdisk0');  
my ($STREAM)      = ('A');  
my ($DATASET)     = ('0');  
my ($STATUS)      = ('open');  
my ($NEVENTS)     = ('0');  
my ($FILESIZE)    = ('0');  
my ($START_TIME)  = ('24-DEC-2006 12.59.59.00');  
my ($STOP_TIME)   = ('25-DEC-2006 01.00.00.00');
my ($CHECKSUM)    = ('0');
my ($CRC)         = ('0');

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if ($#ARGV ==  15)      { $RUNNUMBER   = "$ARGV[0]";
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
			  $CRC         = "$ARGV[15]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Do the update
my $SQLUPDATE = "INSERT INTO CMS_STO_MGR_ADMIN.RUN_FILES (RUNNUMBER,LUMISECTION,INSTANCE,PRODUCER,PATHNAME,FILENAME,HOSTNAME,STREAM,DATASET,STATUS,NEVENTS,FILESIZE,START_TIME,STOP_TIME,CHECKSUM,CRC) VALUES ($RUNNUMBER,$LUMISECTION,$INSTANCE,'$PRODUCER','$PATHNAME','$FILENAME','$HOSTNAME','$STREAM','$DATASET','$STATUS',$NEVENTS,$FILESIZE,'$START_TIME','$STOP_TIME',$CHECKSUM,$CRC)";

my $sth = $dbh->do($SQLUPDATE);
print "$SQLUPDATE \n";

# Disconnect from DB
$dbh->disconnect;
exit 0;
