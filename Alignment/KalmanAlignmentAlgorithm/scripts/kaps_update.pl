#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_update.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Update local kaps database with batch job status
#  
#
#  Usage:
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

read_db();

# initialize FLAG array. Mark jobs we do not have to worry about
@FLAG = -1;
for ($i=0; $i<@JOBID; ++$i) {
  if (         @JOBSTATUS[$i] eq "SETUP"
	       or @JOBSTATUS[$i] eq "DONE"
	       or @JOBSTATUS[$i] eq "FETCH"
	       or @JOBSTATUS[$i] eq "OK"
	       or @JOBSTATUS[$i] eq "ABEND"
	       or @JOBSTATUS[$i] eq "FAIL") {
    @FLAG[$i] = 1;
  }
  else {
    @FLAG[$i] = -1;
  }
}

# first check pending/running jobs shown under job status
@RESULT = `bjobs`;
# always remove the header line
shift @RESULT;
while (@RESULT) {
  $line = shift @RESULT;
  @LINE = split " ",$line;
  # check for job line
  print "line $line\n";
  print "n is $#LINE\n";
 if ($#LINE == 9 || $#LINE == 8) {
    $nn = (@LINE[0] =~ m/(\d+)/);
    $statField = 2;
#    if ($#LINE == 4) {
#      $statField = 4;
#    }
    print "kaps_update: nn is $nn\n";
    if ($nn eq 1) {
      if ($1 eq @LINE[0]) {
	$theId = sprintf "%07d",$1;
	# find the corresponding entry
	$theIndex = -1;
	print "Search index for jobid $theId\n";
	for ($i=0; $i<@JOBID; ++$i) {
	  if (@JOBID[$i] == $theId) {
	    $theIndex = $i;
	  }
	  print "For index $i check jobid @JOBID[$i] result $theIndex\n";
	}
	print "theid $theId i $i theIndex $theIndex\n"; 
	# check if job is already done
	print "FLAG[theIndex] @FLAG[$theIndex]\n";
	if (@FLAG[$theIndex] eq 1) {
	  next;
	}
	# update the status
	if ($theIndex >= 0) {
	  print "update status to @LINE[$statField]\n";
	  @JOBSTATUS[$theIndex] = @LINE[$statField];


	  if (@JOBSTATUS[$theIndex] eq "RUN" || @JOBSTATUS[$theIndex] eq "DONE") {
	    # look for CPU time
	    $result = `bjobs -l $theId`;
	    $nn = ($result =~ m/The CPU time used is +(\d+) +seconds/);
	    if ($nn == 1) {
	      $diff = $1 - @JOBRUNTIME[$theIndex];
	      @JOBRUNTIME[$theIndex] = $1;
	      @JOBHOST[$theIndex] = "+$diff";
	      @JOBINCR[$theIndex] = $diff;
	    }
	    else {
	      @JOBRUNTIME[$theIndex] = 0;
	      @JOBINCR[$theIndex] = 0;
	    }
	  }


	  # flag the index
	  @FLAG[$theIndex] = 1;
	}
      }
    }
  }
}

# loop over remaining jobs to see whether they are done
$theIndex = -1;
for ($i=0; $i<@JOBID; ++$i) {
  $theIndex = $i;
  print " DB job @JOBID[$i] flag @FLAG[$theIndex]\n";
  if (@FLAG[$theIndex] == 1) {
    next;
  }
  # check whether job may be done
  $theBatchDirectory = sprintf "LSFJOB\_%d",@JOBID[$i];
  print "theBatchDirectory $theDirectory\n";
  ## if (-d "LSFJOB\_@JOBID[$i]") {
  ##  print "LSFJOB\_@JOBID[$i] exists\n";
  if (-d $theBatchDirectory) {
    print "Directory $theBatchDirectory exists\n";
    @JOBSTATUS[$theIndex] = "DONE";
  }
}


# check for orphaned jobs
for ($i=0; $i<@JOBID; ++$i) {
  unless (@FLAG[$i] eq 1) {
    unless (@JOBSTATUS[$i] eq "SETUP" 
	    or @JOBSTATUS[$i] eq "DONE" 
	    or @JOBSTATUS[$i] eq "FETCH"
	    or @JOBSTATUS[$i] eq "TIMEL"
	    or @JOBSTATUS[$i] eq "SUBTD") {
      print "Funny entry index $i job @JOBID[$i] status @JOBSTATUS[$i]\n";
    }
  }
}
write_db();

