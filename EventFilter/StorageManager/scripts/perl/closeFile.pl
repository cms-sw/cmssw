#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 31.
# $Id:$
################################################################################

use strict;
use DBI;

################################################################################

sub show_help {  my $exit_status = shift@_;
  print " 
  ############################################################################## 

  Action:
  =======
  Script to do DB operation for Storage Manager upon closing a file
  - update STATUS to closed
  - update STOP_TIME  (default is current time)
  - update PATHNAME

  Syntax:
  =======
  ./closeFile.pl <file-name> [<status> <stop_time> <path_name>]

  - h          to get this help 

  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################

my ($FILENAME,$STATUS,$PATHNAME) = ("dummy","closed","dummy");
my $STOP_TIME = time;

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME  = "$ARGV[0]";}
elsif ($#ARGV ==  1)    { $FILENAME  = "$ARGV[0]"; 
			  $STATUS    = "$ARGV[1]";}
elsif ($#ARGV ==  2)    { $FILENAME  = "$ARGV[0]"; 
			  $STATUS    = "$ARGV[1]"; 
			  $STOP_TIME = "$ARGV[2]";}
elsif ($#ARGV ==  3)    { $FILENAME  = "$ARGV[0]"; 
			  $STATUS    = "$ARGV[1]"; 
			  $STOP_TIME = "$ARGV[2]";
			  $PATHNAME  = "$ARGV[3]";}
else                    { &show_help(1);          }

################################################################################
# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Do the updates
my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.TIER0_INJECTION SET STATUS    = '$STATUS' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);
#printf "$SQLUPDATE \n";

my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.TIER0_INJECTION SET STOP_TIME = '$STOP_TIME' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);
#printf "$SQLUPDATE \n";

if ( $#ARGV ==  3 )
{
    my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.TIER0_INJECTION SET PATHNAME = '$PATHNAME' WHERE FILENAME = '$FILENAME'";
    my $sth = $dbh->do($SQLUPDATE);
    #printf "$SQLUPDATE \n";
}
    
# Disconnect from DB
$dbh->disconnect;
exit 0;
