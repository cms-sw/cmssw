#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg      3-Jul-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.2 $
#     $Date: 2008/04/14 19:19:47 $
#
#  Submit jobs that are setup in local mps database
#  
#
#  Usage:
#
#  mps_fire.pl [-m] [maxjobs]

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$maxJobs = 1;  # by default, fire one job only
$fireMerge = 0;
# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "m") {
      $fireMerge = 1;
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $maxJobs = $arg;
    }
  }
}

read_db();

# build the absolute job directory path (needed by mps_script)
$thePwd = `pwd`;
chomp $thePwd;
$theJobData = "$thePwd/jobData";


if ($fireMerge == 0) {
    # fire the "normal" parallel jobs
    # set the resource string coming from mps.db
    $resources = get_class("mille");
    if ($resources eq "cmscafspec") {  # special cmscaf resource
	print "\nWARNING:\n  Running mille jobs on cmscafspec, intended for pede only!\n\n";
	$resources = "-q cmscaf -R ".$resources;
    } else {
	$resources = "-q ".$resources;
    }

    # set the job name
    $theJobName = "mpalign";
    if ($addFiles ne "") { $theJobName = $addFiles; }

    $nSub = 0;
    for ($i = 0; $i < $nJobs; ++$i) {
	if (@JOBSTATUS[$i] eq "SETUP") {
	    if ($nSub < $maxJobs) {
		# for some reasons LSF wants script with full path
		print "bsub -J $theJobName $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
		$result = `bsub -J $theJobName $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
		print "      $result";
		chomp $result;
		$nn = ($result =~ m/Job \<(\d+)\> is submitted/);
		if ($nn eq 1) {
		    # need standard format for job number
		    @JOBSTATUS[$i] = "SUBTD";
		    ## @JOBID[$i] = $1;
		    @JOBID[$i] = sprintf "%07d",$1;
		    ## print "jobid is @JOBID[$i]\n";
		}
		else {
		    $jid = $i + 1;
		    print "Submission of $jid seems to have failed: $result\n";
		}
		++$nSub;
	    }
	}
    }
}
else {
    # fire the merge job
    print "fire merge\n";
    # set the resource string coming from mps.db
    $resources = get_class("pede");
    if ($resources eq "cmscafspec") {  # special cmscaf resource
	$resources = "-q cmscaf -R ".$resources;
    } else {
	$resources = "-q ".$resources;
    }

    # check whether all other jobs OK
    $mergeOK = 1;
    for ($i = 0; $i < $nJobs; ++$i) {
	if (@JOBSTATUS[$i] ne "OK") {
	    $mergeOK = 0;
	    break;
	}
    }
    
    $i = $nJobs;
    if ((@JOBSTATUS[$i] eq "SETUP") && ($mergeOK == 1)) {
	print "bsub -J almerge $resources $theJobData/@JOBDIR[$i]/theScript.sh\n";
	$result = `bsub -J almerge $resources $theJobData/@JOBDIR[$i]/theScript.sh`;
	print "     $result";
	chomp $result;
	$nn = ($result =~ m/Job \<(\d+)\> is submitted/);
	if ($nn eq 1) {
	    # need standard format for job number
	    @JOBSTATUS[$i] = "SUBTD";
	    ## @JOBID[$i] = $1;
	    @JOBID[$i] = sprintf "%07d",$1;
	    print "jobid is @JOBID[$i]\n";
	}
	else {
	    print "Submission of merge job seems to have failed: $result\n";
	}
    }
    else {
      print "Merge job $i status @JOBSTATUS[$i] not submitted\n";
    }
}
write_db();
