#!/usr/bin/env perl
# $Id: smCleanupFiles.pl,v 1.5 2009/11/16 11:10:40 gbauer Exp $

use strict;
use warnings;
use DBI;
use Getopt::Long;
use File::Basename;

my ($help, $debug, $nothing, $now, $force, $execute, $maxfiles, $fileagemin);
my ($hostname, $filename, $dataset, $stream, $config);
my ($runnumber, $uptorun, $safety, $rmexitcode, $chmodexitcode, );
my ($constraint_runnumber, $constraint_uptorun, $constraint_filename, $constraint_hostname, $constraint_dataset);

sub usage
{
  print "
  ##############################################################################

  Usage $0 [--help] [--debug] [--nothing]  [--now]
  Almost all the parameters are obvious. Non-obvious ones are:
   --now : suppress 'sleep' in delete based on host ID

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
$now        = 0;
$filename   = '';
$dataset    = '';
$uptorun    = 0;
$runnumber  = 0;
$safety     = 100;
$hostname   = '';
$execute    = 1;
$maxfiles   = 1;
$fileagemin = 130;
$force      = 0;
$config     = "/opt/injectworker/.db.conf";

$hostname   = `hostname -s`;
chomp($hostname);

GetOptions(
           "help"          =>\$help,
           "debug"         =>\$debug,
           "nothing"       =>\$nothing,
           "now"           =>\$now,
           "force"         =>\$force,
           "config=s"      =>\$config,
           "hostname=s"    =>\$hostname,
           "run=i"         =>\$runnumber,
           "runnumber=i"   =>\$runnumber,
	   "uptorun=s"	   =>\$uptorun,
	   "filename=s"	   =>\$filename,
	   "dataset=s"	   =>\$dataset,
           "stream=s"      =>\$stream,
           "maxfiles=i"    =>\$maxfiles,
           "fileagemin=i"  =>\$fileagemin
	  );

$help && usage;
if ($nothing) { $execute = 0; $debug = 1; }


#time stagger deletes on the various MAIN SM nodes:
if (!$now) {
    my $deletedelay = 4;
    if( my ( $rack, $node ) = ( $hostname =~ /srv-c2c(0[67])-(\d+)$/i ) ) {
	my $nodesleep = $deletedelay * ( 2 * ( $rack - 6 ) + $node - 12 );
        $debug && print "For node $hostname go to sleep for $nodesleep min\n";
	sleep $nodesleep * 60;
    }
}


my $reader = "xxx";
my $phrase = "xxx";
if(-e $config) {
    eval `su smpro -c "cat $config"`;
} else {
    print "Error: Can not read config file $config, exiting!\n";
    usage();
}

# Look for files in FILES_TRANS_CHECKED - implies closed and safety >= 100 in the old scheme.
# Alternate queries for different values of these? even needed?
# These files need to be in FILES_CREATED and FILES_INJECTED to
# check correct hostname and pathname. They must not be in FILES_DELETED.
my $basesql = "select PATHNAME, CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME, HOSTNAME from CMS_STOMGR.FILES_TRANS_CHECKED inner join " .
               "CMS_STOMGR.FILES_CREATED on CMS_STOMGR.FILES_CREATED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME inner join " .
               "CMS_STOMGR.FILES_INJECTED on CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME=CMS_STOMGR.FILES_INJECTED.FILENAME " .
               "where not exists (select * from CMS_STOMGR.FILES_DELETED " .
                                  "where CMS_STOMGR.FILES_DELETED.FILENAME=CMS_STOMGR.FILES_TRANS_CHECKED.FILENAME)";

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

$debug && print "******BASE QUERY:\n   $myquery,\n";

my $dbi    = "DBI:Oracle:cms_rcms";
my $dbh    = DBI->connect($dbi,$reader,$phrase)
    or die "Can't make DB connection: $DBI::errstr\n";

my $insertDel = $dbh->prepare("insert into CMS_STOMGR.FILES_DELETED (FILENAME,DTIME) VALUES (?,TO_DATE(?,'YYYY-MM-DD HH24:MI:SS'))");
my $sth  = $dbh->prepare($myquery);
$sth->execute() || die "Initial DB query failed: $dbh->errstr\n";

############## Parse and process the result
my $nFiles   = 0;
my $nRMFiles = 0;
my $nRMind   = 0;

my @row;

$debug && print "MAXFILES: $maxfiles\n";

while ( $nFiles<$maxfiles &&  (@row = $sth->fetchrow_array) ) {

    $rmexitcode = 9999;    # Be over-cautious and set a non-zero default
    $debug   && print "       --------------------------------------------------------------------\n";
    $nFiles++;

    # get .ind file name
    my $file =  "$row[0]/$row[1]";
    my $fileIND;
    if ( $file =~ /^(.*)\.(?:dat|root)$/ ) {
        $fileIND = $1 . '.ind';
    }

    if ( -e $file ) {
        my $FILEAGEMIN   =  (time - (stat(_))[9])/60;
        $debug   && print "$FILEAGEMIN $fileagemin\n";

        if ($execute && $FILEAGEMIN > $fileagemin) {
            if ( unlink( $file ) == 1 ) {
                # unlink should return 1: removed 1 file
		$rmexitcode = 0;
            } else {
                print "Removal of $file failed\n";
                $rmexitcode = 9996;
           }
        } elsif (!$execute && $FILEAGEMIN > $fileagemin) {
	    #if we're not executing anything want to fake
	    print "Pretending to remove $file\n";
	    $rmexitcode = 0;
        } elsif ($FILEAGEMIN < $fileagemin) {
            print "File $file too young to die\n";
            $rmexitcode = 9995;
        } else {
            print "This should never happen. File $file has issues!\n";
            $rmexitcode = 9994;
        }
    } elsif ($force) {
        print "File $file does not exist, but force=1, so continue\n";
        $rmexitcode = 0;
    } elsif ( ! -d $row[0] ) {
        print "Path $row[0] does not exist. Are the disks mounted?\n";
        $rmexitcode = 9998;
    } else {
        print "File $file does not exist\n";
        $rmexitcode = 9997;
    }
    #$rmexitcode =0;
    # check file was really removed
    if ($rmexitcode != 0) {
	print "Could not delete file: $file (rmexitcode=$rmexitcode)\n";
    } elsif ( ! -e $file ) {
	$nRMFiles++;
	
	# insert file into deleted db
	$insertDel->bind_param(1,$row[1]);
	$insertDel->bind_param(2,gettimestamp(time));
	$execute && ($insertDel->execute() || die "DB insert into deleted files failed: $insertDel->errstr\n");
	my $delErr = $insertDel->errstr;
	if(defined($delErr)) {
	    print "Delete DB insert produced error: $delErr\n";
	} else {
	    $debug && print "File inserted into deleted DB, rm .ind\n";
	    if ( $execute && -e $fileIND) {
                my $rmIND = `rm -f $fileIND`;
                if (! -e "$fileIND" ) {$nRMind++;}
	    }
	}
    } else {
        print "Unlink returned success, but file $file is still there!\n";
    }


}

#make sure handles are done
$sth->finish();

$dbh->disconnect;

# Only print summary if STDIN is a tty, so not in cron
if( -t STDIN ) {
    print "\n=================> DONE!:\n";
    print ">>BASE QUERY WAS:\n   $myquery,\n";
    print " $nFiles Files Processed\n" .
      " $nRMFiles Files rm-ed\n" .
      " $nRMind ind Files removed\n\n\n";
}

exit 0;
