#!/usr/bin/env perl
# $Id: updateStatus.pl,v 1.1 2007/01/29 13:33:22 klute Exp $
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
  updateStatus.pl <file-name> [<status>]

  - h          to get this help 

  Example:
  ========
  ./updateStatus.pl 'mtcc.00004565.B.testStorageManager_0.0.dat'
  update the status of the file 'mtcc.00004565.B.testStorageManager_0.0.dat' 
  to 'copied'.

  ./updateStatus.pl 'mtcc.00004565.B.testStorageManager_0.0.dat' 'saved'
  update the status of the file 'mtcc.00004565.B.testStorageManager_0.0.dat' 
  to 'saved'.
  
  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME,$STATUS) = ("dummy","copied");

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME = "$ARGV[0]"; }
elsif ($#ARGV ==  1)    { $FILENAME = "$ARGV[0]"; 
			  $STATUS   = "$ARGV[1]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Do the update
my $SQLUPDATE = "UPDATE CMS_STOMGR.TIER0_INJECTION SET STATUS = '$STATUS' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);

# Disconnect from DB
$dbh->disconnect;
exit 0;
