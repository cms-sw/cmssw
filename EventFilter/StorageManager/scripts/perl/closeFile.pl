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
  - update STOP_TIME 

  Syntax:
  =======
  ./closeFile.pl <file-name> [<status> <stop time>]

  - h          to get this help 

  ##############################################################################   
  \n";
  exit $exit_status;
}
################################################################################
# need to get the time and date in "oracle" format
sub CurrentTime
{
    my @date = split(" ",`date`);
    my $month = @date[1];
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime time;
    my $dt = "AM";

    if    ($hour == 0) {$hour = 12;       $dt = "PM";}
    elsif ($hour > 12) {$hour = $hour-12; $dt = "PM";}
		     
    $year=$year+1900;
    $min  = '0' . $min  if ($min < 10);
    $hour = '0' . $hour if ($hour < 10);
    $sec  = '0' . $sec  if ($sec < 10);
    return "$mday-$month-$year $hour.$min.$sec.000000 $dt +00:00";
}

################################################################################

my ($FILENAME,$STATUS) = ("dummy","closed");
my $STOP_TIME = CurrentTime();

if ("$ARGV[0]" eq "-h") { &show_help(0);          }
if    ($#ARGV ==  0)    { $FILENAME  = "$ARGV[0]";}
elsif ($#ARGV ==  1)    { $FILENAME  = "$ARGV[0]"; 
			  $STATUS    = "$ARGV[1]";}
elsif ($#ARGV ==  2)    { $FILENAME  = "$ARGV[0]"; 
			  $STATUS    = "$ARGV[1]"; 
			  $STOP_TIME = "$ARGV[2]";}
else                    { &show_help(1);          }

################################################################################
# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Do the updates
my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET STATUS    = '$STATUS' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);
printf "$SQLUPDATE \n";

my $SQLUPDATE = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET STOP_TIME = '$STOP_TIME' WHERE FILENAME = '$FILENAME'";
my $sth = $dbh->do($SQLUPDATE);
printf "$SQLUPDATE \n";

# Disconnect from DB
$dbh->disconnect;
exit 0;
