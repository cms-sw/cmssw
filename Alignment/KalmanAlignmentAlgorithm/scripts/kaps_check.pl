#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_check.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Check output from jobs that have FETCH status
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
# loop over FETCH jobs
for ($i=0; $i<@JOBID; ++$i) {
  $finished = 0;
  $killed = 0;
  $timeout = 0;
  $cfgerr = 0;
  $emptyDatErr = 0;

  $remark = "";

  if (@JOBSTATUS[$i] eq "FETCH") {

    # open the STDOUT file
    $stdOut = "jobData/@JOBDIR[$i]/STDOUT";
    open STDFILE,"$stdOut";
    $nEvent = -1;
    while ($line = <STDFILE>) {
      if (($line =~ m/CERN report: Job Killed/) eq 1) { $killed = 1;}
      if (($line =~ m/Job finished/) eq 1)  { $finished = 1; }
      if (($line =~ m/connection timed out/) eq 1)  { $timeout = 1; }
      if (($line =~ m/ConfigFileReadError/) eq 1)  { $cfgerr = 1; }
    }
    close STDFILE;

    $kaaOut = sprintf("$mssDir/kaaOutput%03d.root",$i+1);
    $mOutSize = `ls -l $kaaOut | awk '{print \$5}'`;
    # checks that kaaOutput file is not empty
    if ( @JOBDIR[$i] ne "jobm" && !($mOutSize>0) ) {
      $emptyDatErr = 1;
    }

    $okStatus = "OK";
    if ($timeout eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] had connection timed out problem\n";
        $remark = "connection timed out";
    }
    if ($cfgerr eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] had config file error\n";
      $remark = "cfg file error";
      $okStatus = "FAIL";
    }
    if ($killed eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] Job Killed (probably time exceeded)\n";
      $remark = "killed";
      $okStatus = "FAIL";
    }
    if ($emptyDatErr == 1) {
      print "$kaaOut file not found or empty\n";
      $remark = "empty kaaOutput";
      $okStatus = "FAIL"; 
    }

    # print warning line to stdout
    if ($okStatus ne "OK") {
      print "@JOBDIR[$i] @JOBID[$i]  --------  $okStatus\n";
    }
    
    # update number of events
    @JOBSTATUS[$i] = $okStatus;
    @JOBREMARK[$i] = $remark;
  }
}
write_db();
