#!/usr/bin/env perl
# $Id: selectFilesInTimeSpan.pl,v 1.1 2007/01/29 13:33:22 klute Exp $
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
  Script to return file names for all files produced 
  between time a and time b. 
  
  Syntax:
  =======
  selectFilesInTimeSpan.pl <time-stamp> [<time-stamp-b>]

  - h          to get this help
 
  Example:
  ========
  ./selectFilesInTimeSpan.pl '10-NOV-2006 12.00.00.00'
  returns names of all files produced after 11/10/2006 

  ./selectFilesInTimeSpan.pl '10-NOV-2006 12.00.00.00' '12-NOV-2006 12.00.00.00'
  returns names of all files produced between 11/10/2006 and 11/12/2006 
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($TIME_A,$TIME_B) = ("*","31-DEC-2099 12.00.00.00");

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $TIME_A   = "$ARGV[0]"; }
elsif ($#ARGV ==  1)    { $TIME_A   = "$ARGV[0]"; 
			  $TIME_B   = "$ARGV[1]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Prepare sql query
my $SQLQUERY = "SELECT RUNNUMBER, LUMISECTION, INSTANCE, COUNT, TYPE, STREAM, STATUS, SAFETY, NEVENTS, FILESIZE, HOSTNAME, PATHNAME, FILENAME FROM CMS_STOMGR.TIER0_INJECTION WHERE STOP_TIME > '$TIME_A' AND STOP_TIME < '$TIME_B'";
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
