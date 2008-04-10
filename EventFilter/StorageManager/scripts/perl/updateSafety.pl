#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 29.
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
  Script to update the status of a file.
  
  Syntax:
  =======
  updateSafety.pl <file-name> [<safety>]

  - h          to get this help 

  Example:
  ========
  ./updateSafety.pl 'mtcc.00004565.B.testStorageManager_0.0.dat'
  update the status of the file 'mtcc.00004565.B.testStorageManager_0.0.dat' 
  to 0.

  ./updateSafety.pl 'mtcc.00004565.B.testStorageManager_0.0.dat' 1
  update the status of the file 'mtcc.00004565.B.testStorageManager_0.0.dat' 
  to 1.
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME,$SAFETY) = ("dummy",0);

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME = "$ARGV[0]"; }
elsif ($#ARGV ==  1)    { $FILENAME = "$ARGV[0]"; 
			  $SAFETY   = "$ARGV[1]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Do the update
my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET SAFETY = '$SAFETY' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);

# Disconnect from DB
$dbh->disconnect;
exit 0;
