#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_fetch.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Fetch jobs that have DONE status
#  This step is mainly foreseen in case job result files need 
#  to be copied from a spool area.
#  On LSF batch, the job output is already in our directories,
#  hence this function does hardly anything except for calling 
#  kaps_check.pl.
#  
#
#  Usage:
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

$nJobs = 0;
read_db();

# loop over DONE jobs
for ($i=0; $i<@JOBID; ++$i) {
  
  if (@JOBSTATUS[$i] eq "DONE") {
    # move the LSF output
    $theJobDir = "jobData/@JOBDIR[$i]";
    $theBatchDirectory = sprintf "LSFJOB\_%d",@JOBID[$i];
    system "mv  $theBatchDirectory/\* $theJobDir/";
    system "rmdir $theBatchDirectory";

    # update the status
    @JOBSTATUS[$i] = "FETCH";
  }
}
write_db();

system "kaps_check.pl";
