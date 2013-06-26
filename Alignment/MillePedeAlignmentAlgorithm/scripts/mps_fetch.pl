#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     07-Jul-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.4 $
#     $Date: 2012/09/10 15:11:04 $
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

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$nJobs = 0;
read_db();

# loop over DONE jobs
for ($i=0; $i<@JOBID; ++$i) {
  
  if (@JOBSTATUS[$i] =~ /DONE/i) {
    # move the LSF output
    $theJobDir = "jobData/@JOBDIR[$i]";
    $theBatchDirectory = sprintf "LSFJOB\_%d",@JOBID[$i];
    system "mv  $theBatchDirectory/\* $theJobDir/";
    system "rmdir $theBatchDirectory";

    # update the status
    if(@JOBSTATUS[$i] =~ /DISABLED/gi)
      {
        @JOBSTATUS[$i] = "DISABLEDFETCH";
      }
    else
      {
        @JOBSTATUS[$i] = "FETCH";
      }
  }
}
write_db();

system "mps_check.pl";
