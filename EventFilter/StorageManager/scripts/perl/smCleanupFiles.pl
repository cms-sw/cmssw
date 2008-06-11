#!/usr/bin/perl -w
# $Id: smCleanupFiles.pl,v 1.1 2008/06/10 12:10:51 loizides Exp $

use strict;
use DBI;
use Getopt::Long;
use File::Basename;

my ($help, $debug, $nothing, $force, $execute, $maxfiles, $maxfile);
my ($hostname, $filename, $dataset, $stream, $status);
my ($runnumber, $uptorun, $safety, $rmexitcode, $chmodexitcode, );
my ($constraint_runnumber, $constraint_uptorun, $constraint_filename, $constraint_hostname, $constraint_dataset);

sub usage
{
  print " 
  ############################################################################## 
  
  Usage $0 [--help] [--debug] [--nothing] 
  Almost all the parameters are obvious. Non-obvious ones are:

  ##############################################################################   
  \n";
  exit 0;
}

#subroutine for getting formatted time for SQL to_date method
sub gettimestamp($)
{
    my $stime = shift;
    my @ltime = localtime($stime);
    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = @ltime;

    $year += 1900;
    $mon++;

    my $timestr = $year."-";
    if ($mon < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mon . "-";

    if ($mday < 10) {
	$timestr=$timestr . "0";
    }

    $timestr=$timestr . $mday . " " . $hour . ":" . $min . ":" . $sec;
    return $timestr;
}

$help       = 0;
$debug      = 0;
$nothing    = 0;
$filename   = ''; 
$dataset    = '';
$uptorun    = 0;
$runnumber  = 0;
$safety     = 100;
$status     = 'closed';
$hostname   = '';
$rmexitcode = 0;
$execute    = 1;
$maxfiles   = 1;
$force      = 0;

$hostname   = `hostname -s`;
chomp($hostname);

GetOptions(
           "help"          => \$help,
           "debug"         => \$debug,
           "nothing"       => \$nothing,
           "force"         => \$force,
           "hostname=s"    => \$hostname,
           "run=s"         => \$runnumber,
           "runnumber=s"   => \$runnumber,
	   "uptorun=s"	   => \$uptorun,
	   "filename=s"	   => \$filename,
	   "dataset=s"	   => \$dataset,
	   "safety=s"	   => \$safety,
           "stream=s"      => \$stream,
           "status=s"      => \$status,
           "maxfiles=s"    => \$maxfiles,
           "maxfile=s"     => \$maxfiles,
	  );

$help && usage;
if ($nothing) { $execute = 0; $debug = 1; }


# Look for files in FILES_TRANS_CHECKED - implies closed and safety >= 100 in the old scheme. 
# Alternate queries for different values of these? even needed?
# These files need to be in FILES_CREATED and FILES_INJECTED to 
# check correct hostname and pathname. They must not be in FILES_DELETED.
my $basesql = "select PATHNAME, CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME, HOSTNAME from CMS_STOMGR.FILES_TRANS_CHECKED inner join CMS_STOMGR.FILES_CREATED on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME inner join CMS_STOMGR.FILES_INJECTED on CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME " .
"where not exists (select * from CMS_STOMGR.FILES_DELETED where CMS_STOMGR.FILES_DELETED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME)";

# Sorting by time
my $endsql = " order by ITIME";

# Additional constraints
$constraint_runnumber = '';
$constraint_uptorun   = '';
$constraint_filename  = '';
$constraint_hostname  = '';
$constraint_dataset   = '';

if ($runnumber) { $constraint_runnumber = " and RUNNUMBER = $runnumber"; }
if ($uptorun)   { $constraint_uptorun   = " and RUNNUMBER >= $uptorun";  }
if ($filename)  { $constraint_filename  = " and CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME = '$filename'";}
if ($hostname)  { $constraint_hostname  = " and HOSTNAME = '$hostname'";} 
if ($dataset)   { $constraint_dataset   = " and SETUPLABEL = '$dataset'";}

# Compose DB query
my $myquery = '';
$myquery = "$basesql $constraint_runnumber $constraint_uptorun $constraint_filename $constraint_hostname $constraint_dataset $endsql";

$debug && print "******BASE QUERY: \n   $myquery, \n";

my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty")
    or die "Can't make DB connection: $DBI::errstr \n";

my $insertDel = $dbh->prepare("insert into CMS_STOMGR.FILES_DELETED (FILENAME,DTIME) VALUES (?,TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))");
my $sth  = $dbh->prepare($myquery);
$sth->execute() || die "Initial DB query failed: $dbh->errstr \n";

############## Parse and process the result
my $nFiles   = 0;
my $nRMFiles = 0;
my $nRMind   = 0;

my @row;  

$debug && print "MAXFILES: $maxfiles \n";

while ( $nFiles<$maxfiles &&  (@row = $sth->fetchrow_array) ) { 

    $debug   && print "       -------------------------------------------------------------------- \n";
    $nFiles++;

    # get .ind file name
    my $fileIND  =  "$row[0]/$row[1]";
    $fileIND =~ s/\.dat$/\.ind/;
    $fileIND =~ s/\.root$/\.ind/;

    # remove file
    my $CHMODCOMMAND = "sudo chmod 666 $row[0]/$row[1]";
    my $RMCOMMAND    = "rm -f $row[0]/$row[1]";
    $debug   && print "$RMCOMMAND \n";;

    $rmexitcode = 9998;

    if ($execute && -e "$row[0]/$row[1]")
    {
	$chmodexitcode = system($CHMODCOMMAND);
	$rmexitcode    = system($RMCOMMAND);	
	$debug  && print "   ===> rm dat file successful?: $rmexitcode \n";
    } elsif (!$execute && -e "$row[0]/$row[1]") {
	#if we're not executing anything want to fake 
	print "Pretending to remove $row[0]/$row[1] \n";
	$rmexitcode = 0;
    } else {
        if ($force) {
            print "File $row[0]/$row[1] does not exist, but force=1, so continue \n";
            $rmexitcode = 0;
        } else {
            print "File $row[0]/$row[1] does not exist \n";
        }
    }
    $debug && print "\n";
    $rmexitcode =0;
    # check file was really removed
    if ($rmexitcode != 0)
    {  
	print "Can not delete file: $row[0]/$row[1] \n";
    } else {
	$nRMFiles++;
	
	# insert file into deleted db
	$insertDel->bind_param(1,$row[1]);
	$insertDel->bind_param(2,gettimestamp(time));
	$execute && ($insertDel->execute() || die "DB insert into deleted files failed: $insertDel->errstr \n");
	my $delErr = $insertDel->errstr;
	if(defined($delErr)) {
	    print "Delete DB insert produced error: $delErr \n";
	} else {
	    $debug && print "File inserted into deleted DB, rm .ind\n";
	    if ( $execute && -e $fileIND) {
		$CHMODCOMMAND = `sudo chmod 666 $fileIND`;
                my $rmIND = `rm -f $fileIND`;
                if (! -e "$fileIND" ) {$nRMind++;}
	    }
	}
    }
    

}

#make sure handles are done
$sth->finish();

$dbh->disconnect;

print "\n=================> DONE!: \n";
print ">>BASE QUERY WAS: \n   $myquery, \n";
print " $nFiles Files Processed\n" . 
      " $nRMFiles Files rm-ed\n" .
      " $nRMind ind Files removed\n\n\n";

exit 0;
