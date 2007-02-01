#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 31.
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
  Script to find file which are safe to be deleted, update status field 
  and delete files.

  Syntax:
  =======
  ./dummyCleanUp.pl [<hostname> <level>]

  Example:
  ========
  ./dummyCleanUp.pl cmsdisk0 100

  \n";
  exit $exit_status;
}
################################################################################
# Need to get the time and date in "oracle" format
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
# Command line and defaults
my ($HOSTNAME) = ("cmsdisk0");
my ($LEVEL)    = (100);

if ("$ARGV[0]" eq "-h") { &show_help(0); }
if ($#ARGV ==  0)      
{ 
    $HOSTNAME  = "$ARGV[0]";
}
elsif ($#ARGV ==  1)      
{ 
    $HOSTNAME  = "$ARGV[0]";
    $LEVEL     = "$ARGV[1]";
}
else 
{ 
    &show_help(1);          
}

################################################################################
# Connect to DB
my $dbi    = "DBI:Oracle:omds";
my $reader = "cms_sto_mgr";
my $dbh = DBI->connect($dbi,$reader,"qwerty");

# Prepare sql query
my $SQL1 = "SELECT  PATHNAME, FILENAME  FROM CMS_STO_MGR_ADMIN.RUN_FILES WHERE SAFETY >= $LEVEL AND HOSTNAME = '$HOSTNAME' and STATUS = 'closed'";
my $sth  = $dbh->prepare($SQL1);

# Execute the SQL
$sth->execute() || die $dbh->errstr;

# Parse the result
my @row;
while (@row = $sth->fetchrow_array) 
{ 
    #change db status to deleted 
    my $SQL2 = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET STATUS = 'deleted' WHERE FILENAME = '@row[1]'";
    $sth     = $dbh->do($SQL2);

    #change db delete_time to current time
    my $DELETE_TIME = CurrentTime();
    my $SQL3 = "UPDATE CMS_STO_MGR_ADMIN.RUN_FILES SET DELETE_TIME = '$DELETE_TIME' WHERE FILENAME = '@row[1]'";
    $sth     = $dbh->do($SQL3);

    #remove file
    my $RMCOMMAND = "rm @row[0]/@row[1]";
    system($RMCOMMAND);
}

# Disconnect from DB
$dbh->disconnect;
exit 0;
