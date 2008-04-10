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
  Script to return file names of a certain status.
  
  Syntax:
  =======
  selectFilesByStatus.pl <status> 

  - h          to get this help
 
  Example:
  ========
  ./selectFilesByStatus.pl 
  returns names of all files with status='closed'

  ./selectFilesByStatus.pl 'deleted'
  returns names of all files with status='deleted'
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($STATUS) = ("closed");

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $STATUS   = "$ARGV[0]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Prepare sql query
my $SQLQUERY = "SELECT  RUNNUMBER, LUMISECTION, INSTANCE, COUNT, TYPE, STREAM, STATUS, SAFETY, NEVENTS, FILESIZE, HOSTNAME, PATHNAME, FILENAME FROM CMS_STO_MGR_ADMIN.RUN_FILES WHERE STATUS = '$STATUS'";
my $sth = $dbh->prepare($SQLQUERY);

# Execute the SQL
$sth->execute() || die $dbh->errstr;

# Parse the result
my @row;
while (@row = $sth->fetchrow_array) { 
  printf "RUNNUMBER=@row[0]  LUMISECTION=@row[1]  INSTANCE=@row[2]  COUNT=@row[3]  TYPE=@row[4]  STREAM=@row[5]  STATUS=@row[6]  SAFETY=@row[7]  NEVENTS=@row[8]  FILESIZE=@row[9]  HOSTNAME=@row[10]  PATHNAME=@row[11]  FILENAME=@row[12] \n";
}

# Disconnect from DB
$dbh->disconnect;
exit 0;
