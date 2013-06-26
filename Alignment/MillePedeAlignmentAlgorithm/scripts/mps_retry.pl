#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     10-Jul-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.7 $ by $Author: jbehr $
#     $Date: 2012/09/10 15:11:05 $
#
#  Re-Setup failed jobs for resubmission
#  
#
#  Usage:
#
#  mps_retry.pl [job sequence numbers | jobstates]
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

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
	if (@JOBSTATUS[$i] =~ /ABEND/i
	    or @JOBSTATUS[$i] =~ /FAIL/i
	    or @JOBSTATUS[$i] =~ /TIMEL/i
	    or $force == 1) {
          my $cutstatus = $JOBSTATUS[$i];
          $cutstatus =~ s/DISABLED//gi;
	    $stateText = "^$cutstatus\$";
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
    $i = $nJobs; # first non-mille job
    while ( $i < @JOBDIR ) {
	if (@JOBSTATUS[$i] eq "ABEND"
	    or @JOBSTATUS[$i] eq "FAIL"
	    or @JOBSTATUS[$i] eq "TIMEL"
	    or $force == 1) {
	    reScheduleM($i,$refresh);
	    ++$nDone;
	}
	++$i;
    }
}

print "$nDone jobs have been rescheduled\n";


write_db();


sub reSchedule() {
  my $disabled = "";
  $disabled = "DISABLED" if($JOBSTATUS[$_[0]] =~ /DISABLED/ig);
  @JOBSTATUS[$_[0]] = $disabled."SETUP";
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
    print "mps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit\n";
    system "mps_split.pl $infiList $i $nJobs >jobData/$theJobDir/theSplit";
    print "mps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.py $theIsn\n";
    system "mps_splice.pl $cfgTemplate jobData/$theJobDir/theSplit jobData/$theJobDir/the.py $theIsn";
    # create the run script
    print "mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.py jobData/$theJobDir/theSplit $theIsn\n";
    system "mps_script.pl $batchScript  jobData/$theJobDir/theScript.sh $theJobData/$theJobDir the.py jobData/$theJobDir/theSplit $theIsn";
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
    $theJobDir = @JOBDIR[$_[0]];
    $batchScriptMerge = $batchScript . "merge";
    print "mps_merge.pl $cfgTemplate jobData/$theJobDir/alignment_merge.py $theJobData/$theJobDir $nJobs\n";
    system "mps_merge.pl $cfgTemplate jobData/$theJobDir/alignment_merge.py $theJobData/$theJobDir $nJobs";
    # create the merge job script
    print "mps_scriptm.pl $batchScriptMerge jobData/$theJobDir/theScript.sh $theJobData/$theJobDir alignment_merge.py $nJobs\n";
    system "mps_scriptm.pl $batchScriptMerge jobData/$theJobDir/theScript.sh $theJobData/$theJobDir alignment_merge.py $nJobs";
  }
  print "ReSchedule @JOBDIR[$_[0]]\n";
}
