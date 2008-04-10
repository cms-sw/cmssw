#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     10-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.14 $
#     $Date: 2008/03/25 16:15:57 $
#
#  Re-Setup failed jobs for resubmission
#  
#
#  Usage:
#
#  mps_retry.pl [job sequence numbers] [jobstates] [replacement class]
#

use lib './mpslib';
use Mpslib;

$sdir = get_sdir();

@MODSTATES = ();
@MODJOBS = ();
$modClass = "";
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

    # array of valid job queues
    push @ARR,"8nm","1nh","8nh","1nd","2nd","1nw","2nw","cmsalca","cmscaf";
    $string = "^$arg\$";
    $nn = grep /$string/,@ARR;
    # print "nn is $nn \n";
    if ( (grep /$string/,@ARR) > 0) {
      print "Parameter $arg interpreted as job class\n";
      $modClass = $arg;
    }
    elsif (($arg =~ m/\d+/) eq 1) {
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

# default modifier settings
if ($modClass eq "") {
  $modClass = $class;  # re-use previous class
}


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
        or @JOBSTATUS[$i] eq "OK"
	or @JOBSTATUS[$i] eq "TIMEL") {
	reScheduleM($i,$refresh);
	++$nDone;
    }
}

if (modClass ne $class) {
  $class = $modClass;
  print "Changed job class to $class\n";
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
    print "$sdir/mps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit\n";
    system "$sdir/mps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit";
    print "$sdir/mps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.cfg $theIsn\n";
    system "$sdir/mps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.cfg $theIsn";
    # create the run script
    print "$sdir/mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.cfg jobData/$theJobDir/theSplit $theIsn\n";
    system "$sdir/mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.cfg jobData/$theJobDir/theSplit $theIsn";
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
    print "$sdir/mps_merge.pl $cfgTemplate jobData/jobm/alignment_merge.cfg $theJobData/jobm $nJobs\n";
    system "$sdir/mps_merge.pl $cfgTemplate jobData/jobm/alignment_merge.cfg $theJobData/jobm $nJobs";
    # create the merge job script
    print "$sdir/mps_scriptm.pl $batchScriptMerge jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.cfg $nJobs\n";
    system "$sdir/mps_scriptm.pl $batchScriptMerge jobData/jobm/theScript.sh $theJobData/jobm alignment_merge.cfg $nJobs";
  }
  print "ReSchedule @JOBDIR[$_[0]]\n";
}
