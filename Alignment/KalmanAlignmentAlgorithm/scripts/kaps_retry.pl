#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_retry.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Re-Setup failed jobs for resubmission
#  
#
#  Usage:
#
#  kaps_retry.pl [job sequence numbers | jobstates]
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

@MODSTATES = ();
@MODJOBS = ();
$refresh = "no";
$retryMerge = 0;
$force = 0;

# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "m") {
      $retryMerge = 1;
      print "option sets retryMerge to $retryMerge\n";
    }
    elsif ($arg =~ "d") {
      $localdir = 1;
    }
    elsif ($arg =~ "r") {
      $refresh = "yes";
      print "refresh set to $refresh\n";
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    elsif ($arg =~ "f") {
      $force = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;

    if (($arg =~ m/\d+/) eq 1) {
      print "Parameter $arg interpreted as job number\n";
      push @MODJOBS,$arg;
    }
    elsif (($arg =~ m/[A-Z]+/) eq 1) {
      print "Parameter $arg interpreted as job state\n";
      push @MODSTATES,$arg;
    }
    else {
      print "Parameter $arg not recognized\n";
    }
  }
}

read_db();

# counter for rescheduled jobs
$nDone = 0;

if ($retryMerge != 1) {
    # loop over all jobs
##    for ($i=0; $i<@JOBID; ++$i) {
    for ($i=0; $i<$nJobs; ++$i) {
	if (@JOBSTATUS[$i] eq "ABEND"
	    or @JOBSTATUS[$i] eq "FAIL"
	    or @JOBSTATUS[$i] eq "TIMEL"
	    or $force == 1) {
	    $stateText = "^@JOBSTATUS[$i]\$";
	    $theNum = $i + 1;
	    $jobText = "^$theNum\$";
	    if ( ( (grep /$stateText/,@MODSTATES) > 0) || (grep /$jobText/,@MODJOBS) > 0) {
		reSchedule($i,$refresh);
		++$nDone;
	    }
	}
    }
}
else {
    # retry only the merge job
    $i = $nJobs;
    if (@JOBSTATUS[$i] eq "ABEND"
	or @JOBSTATUS[$i] eq "FAIL"
	or @JOBSTATUS[$i] eq "TIMEL"
	or $force == 1) {
	reScheduleM($i,$refresh);
	++$nDone;
    }
}

print "$nDone jobs have been rescheduled\n";


write_db();


sub reSchedule() {
  @JOBSTATUS[$_[0]] = "SETUP";
  @JOBID[$_[0]] = 0;
  @JOBHOST[$_[0]] = "";
  ++@JOBNTRY[$_[0]];
  if ($_[1] eq "yes") {
    # re-create the split card files
    $thePwd = `pwd`;
    chomp $thePwd;
    $theJobData = "$thePwd/jobData";
    $theJobDir = sprintf "job%03d",$_[0]+1;
    $theIsn = sprintf "%03d",$i;
    print "kaps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit\n";
    system "kaps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit";
    print "kaps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the_cfg.py $theIsn\n";
    system "kaps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the_cfg.py $theIsn";
    # create the run script
    print "kaps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the_cfg.py jobData/$theJobDir/theSplit $theIsn\n";
    system "kaps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the_cfg.py jobData/$theJobDir/theSplit $theIsn";
  }
  print "ReSchedule @JOBDIR[$_[0]]\n";
}

sub reScheduleM() {
  @JOBSTATUS[$_[0]] = "SETUP";
  @JOBID[$_[0]] = 0;
  @JOBHOST[$_[0]] = "";
  ++@JOBNTRY[$_[0]];
  if ($_[1] eq "yes") {
    # re-create the split card files
    $thePwd = `pwd`;
    chomp $thePwd;
    $theJobData = "$thePwd/jobData";
    $theJobDir = "jobm";
    $batchScriptMerge = $batchScript . "merge";
    print "kaps_merge.pl $cfgTemplate jobData/jobm/alignment_merge.cfg $theJobData/jobm $nJobs\n";
    system "kaps_merge.pl $cfgTemplate jobData/jobm/alignment_merge.cfg $theJobData/jobm $nJobs";
    # create the merge job script
    print "kaps_scriptm.pl $batchScriptMerge jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.cfg $nJobs\n";
    system "kaps_scriptm.pl $batchScriptMerge jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.cfg $nJobs";
  }
  print "ReSchedule @JOBDIR[$_[0]]\n";
}
