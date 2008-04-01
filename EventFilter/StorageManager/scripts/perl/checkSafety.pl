#!/usr/bin/env perl
# $Id: checkSafety.pl,v 1.1 2007/01/29 13:33:22 klute Exp $
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
  Script to return the safety of a file 
  
  Syntax:
  =======
  checkSafety.pl <file-name> 

  - h          to get this help
 
  Example:
  ========
  ./checkSafety.pl 'mtcc.00004565.B.testStorageManager_0.0.dat'
  returns safety of  mtcc.00004565.B.testStorageManager_0.0.dat
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME) = ("dummy");

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if ($#ARGV ==  0)       { $FILENAME = "$ARGV[0]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Prepare sql query
my $SQLQUERY = "SELECT SAFETY FROM CMS_STOMGR.TIER0_INJECTION WHERE FILENAME = '$FILENAME' ";
my $sth = $dbh->prepare($SQLQUERY);

# Execute the SQL
$sth->execute() || die $dbh->errstr;

# Parse the result
my @row;
while (@row = $sth->fetchrow_array) { 
  printf "@row[0] @row[1] @row[2] @row[3]\n";
}

# Disconnect from DB
$dbh->disconnect;
exit 0;
