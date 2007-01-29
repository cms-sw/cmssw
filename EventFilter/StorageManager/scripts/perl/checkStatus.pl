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
  Script to return the status of a file 
  
  Syntax:
  =======
  checkStatus.pl <file-name> 

  - h          to get this help
 
  Example:
  ========
  ./checkStatus.pl 'mtcc.00004565.B.testStorageManager_0.0.dat'
  returns status of  mtcc.00004565.B.testStorageManager_0.0.dat
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME) = ("dummy");

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME = "$ARGV[0]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Prepare sql query
my $SQLQUERY = "SELECT STATUS FROM CMS_STO_MGR_ADMIN.RUN_FILES WHERE FILENAME = '$FILENAME' ";
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
