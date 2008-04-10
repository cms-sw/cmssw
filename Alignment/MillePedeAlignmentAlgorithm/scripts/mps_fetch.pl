#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     07-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.14 $
#     $Date: 2008/03/25 16:15:57 $
#
#  Fetch jobs that have DONE status
#  This step is mainly foreseen in case job result files need 
#  to be copied from a spool area.
#  On LSF batch, the job output is already in our directories,
#  hence this function does hardly anything except for calling 
#  mps_check.pl.
#  
#
#  Usage:
#

use lib './mpslib';
use Mpslib;

$sdir = get_sdir();

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

system "$sdir/mps_check.pl";
