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
  Script to delete a file from the run_files table.

  Syntax:
  =======
  deleteFile.pl <file-name> 

  - h          to get this help

  Example:
  ========
  ./deleteFile.pl test
  deletes from entry from the run_files table where the file name is equal test

  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME)    = ('test');

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME = "$ARGV[0]"; }
else                    { &show_help(1);          }

# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr_admin";
my $dbh    = DBI->connect($dbi,$reader,"qwerty");

# Do the update
my $SQLUPDATE = "DELETE FROM CMS_STO_MGR_ADMIN.RUN_FILES WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);
print "$SQLUPDATE \n";

# Disconnect from DB
$dbh->disconnect;
exit 0;
