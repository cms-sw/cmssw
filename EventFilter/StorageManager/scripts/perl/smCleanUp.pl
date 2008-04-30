#!/usr/bin/perl -w
# $Id:$

use strict;
use DBI;
use Getopt::Long;
use File::Basename;

my ($help, $debug, $nothing, $execute, $maxfiles, $maxfile);
my ($hostname, $filename, $dataset, $stream, $status);
my ($runnumber, $uptorun, $safety, $rmexitcode, $chmodexitcode);
my ($constraint_runnumber, $constraint_uptorun, $constraint_filename);
my ($constraint_hostname, $constraint_dataset);

################################################################################
sub usage
{
  print " 
  ############################################################################## 
  
  Usage $0 
           [--help] [--debug] [--nothing] [--hostname] 
           [--runnumber] [--uptorun] [--filename] [--dataset] 
           [--safety] [--stream] [--status] [--maxfiles]

  ##############################################################################   
  \n";
  exit 0;
}

################################################################################

$help      = 0;
$debug     = 0;
$nothing   = 0;
$filename  = ''; 
$dataset   = '';
$uptorun   = 0;
$runnumber = 0;
$safety    = 100;
$status    = 'closed';
$hostname  = '';
$rmexitcode= 0;
$execute   = 1;
$maxfiles  = 1;

#Default is to use host where script is being run (can be overridden by command line):
$hostname = `hostname -s`;
chomp($hostname);

GetOptions(
           "help"          => \$help,
           "debug"         => \$debug,
           "nothing"       => \$nothing,
           "hostname=s"    => \$hostname,
           "runnumber=s"   => \$runnumber,
	   "uptorun=s"	   => \$uptorun,
	   "filename=s"	   => \$filename,
	   "dataset=s"	   => \$dataset,
	   "safety=s"	   => \$safety,
           "stream=s"      => \$stream,
           "status=s"      => \$status,
           "maxfiles=s"    => \$maxfiles,
	  );

$help && usage;
if ($nothing) { $execute = 0; $debug = 1; }

################################################################################

# Base Query
my $basesql= "select PATHNAME, FILENAME from CMS_STOMGR.TIER0_INJECTION where SAFETY >= $safety and STATUS = '$status'";

# Sorting by time
my $endsql = " order by STOP_TIME";

# Additional constraints
$constraint_runnumber = '';
$constraint_uptorun   = '';
$constraint_filename  = '';
$constraint_hostname  = '';
$constraint_dataset   = '';

if ($runnumber) { $constraint_runnumber = " and RUNNUMBER = $runnumber"; }
if ($uptorun)   { $constraint_uptorun   = " and RUNNUMBER >= $uptorun";  }
if ($filename)  { $constraint_filename  = " and FILENAME = '$filename'";}
if ($hostname)  { $constraint_hostname  = " and HOSTNAME = '$hostname'";} 
if ($dataset)   { $constraint_dataset   = " and DATASET = '$dataset'";}

# Compose DB query
my $myquery = '';
$myquery = "$basesql $constraint_runnumber $constraint_uptorun $constraint_filename $constraint_hostname $constraint_dataset $endsql";

$debug && print "******BASE QUERY: \n   $myquery, \n";

# Connect to DB
my $dbi    = "DBI:Oracle:cms_rcms";
my $reader = "CMS_STOMGR_W";
my $dbh    = DBI->connect($dbi,$reader,"qwerty")
    or die "Can't make DB connection: $DBI::errstr \n";

# Prepare queries
my $update_stat    = $dbh->prepare("update CMS_STOMGR.RUN_FILES set STATUS = 'deleted' where FILENAME = ?");
my $update_time    = $dbh->prepare("update CMS_STOMGR.RUN_FILES set DELETE_TIME = ? where FILENAME = ?");
my $delete_file    = $dbh->prepare("delete from CMS_STOMGR.TIER0_INJECTION where FILENAME = ?");
my $debug_runfiles = $dbh->prepare("select HOSTNAME, PATHNAME, FILENAME, STATUS, SAFETY, DELETE_TIME  from CMS_STOMGR.RUN_FILES where FILENAME = ?");
my $debug_inject   = $dbh->prepare("select HOSTNAME, PATHNAME, FILENAME, STATUS, SAFETY, DELETE_TIME  from CMS_STOMGR.TIER0_INJECTION where FILENAME = ?");

# Prepare and execute myquery
my $sth  = $dbh->prepare($myquery);
$sth->execute() || die "Initial DB query failed: $dbh->errstr \n";

############## Parse and process the result
my $nFiles   = 0;
my $nRMFiles = 0;
my $nDeltDB  = 0;
my $nRMind   = 0;
my $nRMsmry  = 0;
my $nRMtxt   = 0;

my @row;  
my  @outrow;

print "$0 info: MAXFILES=$maxfiles \n";

while ($nFiles<$maxfiles &&  (@row = $sth->fetchrow_array)) { 
  $debug   && print "       -------------------------------------------------------------------- \n";
  $nFiles++;

  #convert *.dat file name being processed into its *.ind and *.smry partners
  #transferred files are allowed to be either *.dat or *.root
  my $fileIND  =  "$row[0]/$row[1]";
  $fileIND =~ s/\.dat$/\.ind/;
  $fileIND =~ s/\.root$/\.ind/;

  my $fileTXT = "$row[0]/$row[1]"; 
  $fileTXT =~ s/\.dat$/\.txt/;
  $fileTXT =~ s/\.root$/\.txt/;

  my @splitAnswr  = split(m/\// , $row[0]);
  my $dirtop    =  $splitAnswr[1];
  my $areatype  =  $splitAnswr[2];
  my $fileSMRY =  "/$splitAnswr[1]/$areatype/mbox/$row[1]";
  $fileSMRY =~ s/\.dat$/\.smry/;
  $fileSMRY =~ s/\.root$/\.smry/;

  #remove file
  my $CHMODCOMMAND = "sudo chmod 666 $row[0]/$row[1]";
  my $RMCOMMAND    = "rm -f $row[0]/$row[1]";
  $debug   && print "$RMCOMMAND \n";

  if ($execute && -e  "$row[0]/$row[1]") {
    $chmodexitcode = system($CHMODCOMMAND);
    $rmexitcode    = system($RMCOMMAND);	
    $debug  && print "   ===> rm dat file successful?: $rmexitcode \n";
  }
  print "\n";

  # check file was really removed
  if ($rmexitcode==0 || $execute==0) { 
    $nRMFiles++;

    if($debug){
      $debug_runfiles->bind_param(1,$row[1]);
      $debug_runfiles->execute();
      @outrow = $debug_runfiles->fetchrow_array;
      if( !(defined  $outrow[5]) ){$outrow[5]= " ? " };
      print  "PRE-RUNFILESDB: $outrow[0]||$outrow[1]||$outrow[2]||$outrow[3]|| $outrow[4] || $outrow[5] | ","\n \n  >> DB actions: \n";
    }

    #change db status to deleted 
    $update_stat->bind_param(1,$row[1]);
    my $SQL2 = "update CMS_STOMGR.RUN_FILES set STATUS = 'deleted' where FILENAME = '$row[1]'";
    $debug   && print $SQL2 , "\n";
    $execute && ($update_stat->execute() || die "DB UPDATE-status failed: $dbh->errstr \n");

    #change db delete_time to current time
    my $DELETE_TIME = time;
    $update_time->bind_param(1,$DELETE_TIME);
    $update_time->bind_param(2,$row[1]);
    my $SQL3 = "update CMS_STOMGR.RUN_FILES set DELETE_TIME = '$DELETE_TIME' where FILENAME = '$row[1]'";
    $debug   && print $SQL3 , "\n";
    $execute && ($update_time->execute() || die "DB UPDATE-time failed: $dbh->errstr \n");

    if($debug){
      $debug_runfiles->bind_param(1,$row[1]);
      $debug_runfiles->execute();
      @outrow = $debug_runfiles->fetchrow_array;
      if( !(defined  $outrow[5]) ){$outrow[5]= " ? " };
      print  "UPDATD ENTRY: $outrow[0]||$outrow[1]||$outrow[2]||$outrow[3]||$outrow[4]|| $outrow[5] | \n\n ";
    }

    if($debug){
      $debug_inject->bind_param(1,$row[1]);
      $debug_inject->execute();
      @outrow = $debug_inject->fetchrow_array;
      if( !(defined  $outrow[5]) ){$outrow[5]= " ? " };
      print "PRE-INJCT delete:\n  $outrow[0] || $outrow[1] ||  $outrow[2] ||  $outrow[3] ||  $outrow[4]||  $outrow[5] | ","\n\n";
    }

    #delete entry from db
    my $SQL4 = "delete from CMS_STOMGR.TIER0_INJECTION where FILENAME = '$row[1]'";
    $delete_file->bind_param(1,$row[1]);
    $debug   && print $SQL4 , "\n";
    $execute && $delete_file->execute();
    my $delete_err = $delete_file->errstr;
	
    #if deleted entry, rm *.ind and *smry files
    if( !(defined $delete_err) ) {
      $nDeltDB++;
      $debug   && print  ">> ...File DELETED from DB, now rm *.ind and *.smry files\n";
      if( -e "$fileIND"){
        $CHMODCOMMAND = "sudo chmod 666 $fileIND";
        my $rmIND = `rm -f $fileIND`;
        if (! -e "$fileIND" ) {$nRMind++;}
      }
      if( -e "$fileSMRY"){
        $CHMODCOMMAND = "sudo chmod 666 $fileSMRY";
        my $rmSMRY = `rm $fileSMRY`;
        if (! -e "$fileSMRY") {$nRMsmry++;}
      }
      if( -e "$fileTXT"){
	$CHMODCOMMAND = "sudo chmod 666 $fileTXT";
        my $rmTXT = `rm $fileTXT`;
        if (! -e "$fileTXT" ) {$nRMtxt++;}
      }
    } else {
      print "DB delete ERRORSTRING:  $delete_err \n";
    }

    if($debug){
      $debug_inject->bind_param(1,$row[1]);
      $debug_inject->execute();
      @outrow = $debug_inject->fetchrow_array;
      if( defined $outrow[0]){
        print  "POST-INJCT: $outrow[0] || $outrow[1] ||  $outrow[2] ||  $outrow[3] | ","\n\n";
      }
    }
  } else {
    print "Can not delete file: $row[0]/$row[1] \n";
  }

  print "=================> $nFiles  Files;  $nRMFiles rm-ed Files;  $nDeltDB  DB-Deletes;   $nRMind rmv-ed ind Files;  $nRMsmry rmv-ed smy Files;  $nRMtxt rmv-ed txt Files\n\n";
}

#put in "finish" calls to avoid "disconnect invalidates 1 active statement handle" error
#do not understand why this might be needed? (could it be cuz on uninit DSB entry?
$sth->finish();
$debug_runfiles ->finish();

# Disconnect from DB
$dbh->disconnect;

print "\n=================> DONE!: \n";
print ">>BASE QUERY WAS: \n   $myquery, \n";
print " $nFiles  Files Processed\n $nRMFiles  Files rm-ed\n $nDeltDB  DB-Deletes\n $nRMind  ind  Files rmv-ed\n $nRMsmry  smry Files rmv-ed\n $nRMtxt  txt-META Files rmv-ed \n\n";

exit 0;
