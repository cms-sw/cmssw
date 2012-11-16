#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     09-Jul-2007
#     A. Parenti, DESY Hamburg    24-Apr-2008
#     $Revision: 1.31 $ by $Author: jbehr $
#     $Date: 2012/09/10 15:11:04 $
#
#  Check output from jobs that have FETCH status
#  
#
#  Usage:
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

read_db();

my @cmslsoutput = `cmsLs -l $mssDir`;

# loop over FETCH jobs
for ($i=0; $i<@JOBID; ++$i) {
  $batchSuccess = 0;
  $batchExited = 0;
  $finished = 0;
  $endofjob = 0;
  $eofile = 1;  # do not deal with timel yet
  $timel = 0;
  $killed = 0;
  $ioprob = 0;
  $fw8001 = 0;
  $tooManyTracks = 0;
  $segviol = 0;
  $rfioerr = 0;
  $quota = 0;
  $nEvent = 0;
  $cputime = -1;
  $pedeAbend = 0;
  $pedeLogErr = 0;
  $pedeLogWrn = 0;
  $exceptionCaught = 0;
  $timeout = 0;
  $cfgerr = 0;
  $emptyDatErr = 0;
  $emptyDatOnFarm = 0;
  $cmdNotFound = 0;
  $insuffPriv = 0;
  $quotaspace = 0;

  $pedeLogErrStr = "";
  $pedeLogWrnStr = "";
  $remark = "";

  my $disabled = "";
  $disabled = "DISABLED" if( $JOBSTATUS[$i] =~ /DISABLED/gi);
  $JOBSTATUS[$i] =~ s/DISABLED//i;
  if (@JOBSTATUS[$i] eq "FETCH") {

    # open the STDOUT file
    $stdOut = "jobData/@JOBDIR[$i]/STDOUT";
    open STDFILE,"$stdOut";
    # scan records in input file
    while ($line = <STDFILE>) {
      if (($line =~ m/Unable to access quota space/) eq 1) { $quotaspace = 1;}
      if (($line =~ m/Unable to get quota space/) eq 1) { $quotaspace = 1;}
      if (($line =~ m/CERN report: Job Killed/) eq 1) { $killed = 1;}
      if (($line =~ m/Job finished/) eq 1)  { $finished = 1; }
      if (($line =~ m/connection timed out/) eq 1)  { $timeout = 1; }
      if ($line =~ m/This job used .+?(\d+) NCU seconds/ eq 1) {
	$ncuFactor = 3; # this factor is most probably out-of-date...
	$cputime = $1 / $ncuFactor;
	# print "Set cpu to $cputime\n";
      }
      if (($line =~ m/ConfigFileReadError/) eq 1)  { $cfgerr = 1; }
      if (($line =~ m/0 bytes transferred/) eq 1)  { $emptyDatOnFarm = 1; }
      if (($line =~ m/command not found/) eq 1)  { $cmdNotFound = 1; }
# AP 26.11.2009 Insufficient privileges to rfcp files
      if (($line =~ m/stage_put: Insufficient user privileges/) eq 1)  { $insuffPriv = 1; }

    }
    close STDFILE;
    # gzip it afterwards:
    print "gzip -f $stdOut\n";
    system "gzip -f $stdOut";
    
    # GF: This file is not produced (anymore...)
    $eazeLog = "jobData/@JOBDIR[$i]/cmsRun.out";
    # open the input file
    open INFILE,"$eazeLog";
    # scan records in input file
    while ($line = <INFILE>) {
      # check if end of file has been reached
      if (($line =~ m/\<StorageStatistics\>/) eq 1) { $eofile = 1;}
      if (($line =~ m/Time limit reached\./) eq 1) { $timel = 1;}
      if (($line =~ m/gives I\/O problem/) eq 1) { $ioprob = 1;}
      if (($line =~ m/FrameworkError ExitStatus=\"8001\"/) eq 1) { $fw8001 = 1;}
      if (($line =~ m/too many tracks/) eq 1) { $tooManyTracks = 1;}
      if (($line =~ m/segmentation violation/) eq 1) { $segviol = 1;}
      if (($line =~ m/failed RFIO error/) eq 1) { $rfioerr = 1;}
      if (($line =~ m/Request exceeds quota/) eq 1) { $quota = 1;}
    }
    close INFILE;

    # if there is an alignment.log[.gz] file, check it as well
    $eazeLog = "jobData/@JOBDIR[$i]/alignment.log";
    $logZipped = "no";
    if (-r $eazeLog.".gz") {
      system "gunzip ".$eazeLog.".gz";
      $logZipped = "true";
    }
    if (-r $eazeLog) {
      # open the input file
      open INFILE,"$eazeLog";
      # scan records in input file
      while ($line = <INFILE>) {
	# check if end of file has been reached
	if (($line =~ m/\<StorageStatistics\>/) eq 1) { $eofile = 1;}
	if (($line =~ m/EAZE\. Time limit reached\./) eq 1) { $timel = 1;}
	if (($line =~ m/GAF gives I\/O problem/) eq 1) { $ioprob = 1;}
	if (($line =~ m/FrameworkError ExitStatus=\"8001\"/) eq 1) { $fw8001 = 1;}
	if (($line =~ m/too many tracks/) eq 1) { $tooManyTracks = 1;}
	if (($line =~ m/segmentation violation/) eq 1) { $segviol = 1;}
	if (($line =~ m/failed RFIO error/) eq 1) { $rfioerr = 1;}
	if (($line =~ m/Request exceeds quota/) eq 1) { $quota = 1;}
	# check for newer (e.g. CMSSW_5_1_X) and older CMSSW:
	if (($line =~ m/Fatal Exception/) eq 1 || ($line =~ m/Exception caught in cmsRun/)) { $exceptionCaught = 1;}
# AP 07.09.2009 - Check that the job got to a normal end
	if (($line =~ m/AlignmentProducer::endOfJob()/) eq 1) { $endofjob = 1;}
	if (($line =~ m/FwkReport            -i main_input:sourc/) eq 1) {
	  @array = split(' ',$line);
	  $nEvent = $array[5];
	}
	if ($nEvent==0 && ($line =~ m/FwkReport            -i PostSource/) eq 1) {
	  @array = split(' ',$line);
	  $nEvent = $array[5];
        }
	if ($nEvent==0 && ($line =~ m/FwkReport            -i AfterSource/) eq 1) {
# AP 31.07.2009 - To read number of events in CMSSW_3_2_2_patch2
	  @array = split(' ',$line);
	  $nEvent = $array[5];
        }

      }
      close INFILE;
      if ($logZipped eq "true") {
	system "gzip $eazeLog";
      }
    } else {
      print "mps_check.pl cannot find $eazeLog to test\n";
# AP 07.09.2009 - The following check cannot be done: set to 1 to avoid fake error type
      $endofjob = 1;
    }

    # for mille jobs checks that milleBinary file is not empty
    if ( $i < $nJobs ) { # mille job!
      my $milleOut = sprintf("milleBinary%03d.dat",$i+1);
      #$mOutSize = `nsls -l $mssDir | grep $milleOut | head -1 | awk '{print \$5}'`;
      #$mOutSize = `cmsLs -l $mssDir | grep $milleOut | head -1 | awk '{print \$2}'`;
      my $mOutSize = 0;
      foreach my $line (@cmslsoutput)
        {
          if($line =~ /$milleOut/)
            {
              my @columns = split " ", $line;
              $mOutSize = $columns[1];
            }
        }
      if ( !($mOutSize>0) ) {
	$emptyDatErr = 1;
      }
    } else { # merge jobs 
      # additional checks for merging job
      # if there is a pede.dump file, check it as well
      $eazeLog = "jobData/@JOBDIR[$i]/pede.dump";
      if (-r $eazeLog.".gz") { # or is it zipped?
	  # unzip - but clean before and tmp
	  # FIXME: use http://perldoc.perl.org/File/Temp.html, not /tmp/pede.dump!!
	  system "rm -f /tmp/pede.dump; gunzip -c ".$eazeLog.".gz > /tmp/pede.dump";
	  $eazeLog="/tmp/pede.dump";
      }
      if (-r $eazeLog) {
	# open the input file
	open INFILE,"$eazeLog";
	# scan records in input file
	$pedeAbend = 1;
	while ($line = <INFILE>) {
	  # check if pede has reached its normal end
#	  if (($line =~ m/Millepede II ending/) eq 1) { $pedeAbend = 0;}
	  if (($line =~ m/Millepede II.* ending/) eq 1) { $pedeAbend = 0;}
	}
	# clean up if needed FIXME if using proper perl File/Temp!
	system "rm /tmp/pede.dump" if ($eazeLog eq "/tmp/pede.dump");
      } else {
	print "mps_check.pl cannot find $eazeLog to test\n";
      }

      # if there is a millepede.log file, check it as well
      $eazeLog = "jobData/@JOBDIR[$i]/millepede.log";
      $logZipped = "no";
      if (-r $eazeLog.".gz") {
        system "gunzip ".$eazeLog.".gz";
        $logZipped = "true";
      }
      if (-r $eazeLog) {
      # open the input file
        open INFILE,"$eazeLog";
      # scan records in input file
        while ($line = <INFILE>) {
# Checks for Pede Errors:
	  if (($line =~ m/step no descending/) eq 1) {$pedeLogErr = 1; $pedeLogErrStr .= $line;}
	  if (($line =~ m/Constraint equation discrepancies:/) eq 1) {$pedeLogErr = 1; $pedeLogErrStr .= $line;}
# AP 07.09.2009 - Checks for Pede Warnings:
	  if (($line =~ m/insufficient constraint equations/) eq 1) {$pedeLogWrn = 1; $pedeLogWrnStr .= $line;}
        }
        close INFILE;
        if ($logZipped eq "true") {
	  system "gzip $eazeLog";
        }
      } else {
        print "mps_check.pl cannot find $eazeLog to test\n";
      }
    }

    $farmhost = " ";

    $okStatus = "OK";
    unless ($eofile eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] did not reach end of file\n";
      $okStatus = "ABEND";
    }
    if ($quotaspace eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] had quota space problem\n";
      $okStatus = "FAIL";
      $remark = "eos quota space problem";
    }
    if ($ioprob eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] had I/O problem\n";
      $okStatus = "FAIL";
    }
    if ($fw8001 eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] had Framework error 8001 problem\n";
      $remark = "fwk error 8001";
      $okStatus = "FAIL";
    }
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
    if ($timel eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] ran into time limit\n";
      $okStatus = "TIMEL";
    }
    if ($tooManyTracks eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] too many tracks\n";
    }
    if ($segviol eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] SEGVIOL encountered\n";
      $remark = "seg viol";
      $okStatus = "FAIL";
    }
    if ($rfioerr eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] RFIO error encountered\n";
      $remark = "rfio error";
      $okStatus = "FAIL";
    }
    if ($quota eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] Request exceeds quota\n";
    }
    if ($exceptionCaught eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] Exception caught in cmsrun\n";
	$remark = "Exception caught";
	$okStatus = "FAIL";
    }
    if ($emptyDatErr == 1) {
      print "milleBinary???.dat file not found or empty\n";
      $remark = "empty milleBinary";
      if ( $emptyDatOnFarm > 0 ) {
	my $num=$i+1;
	print "...but already empty on farm so OK (or check job $num yourself...)\n";
      } else {
	$okStatus = "FAIL";
      }
    }

    if ($cmdNotFound eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] Command not found\n";
	$remark = "cmd not found";
	$okStatus = "FAIL";
    }
    if ($insuffPriv eq 1) {
        print "@JOBDIR[$i] @JOBID[$i] Insufficient privileges to rfcp files\n";
        $remark = "Could not rfcp files";
        $okStatus = "FAIL";
    }

    if ($pedeAbend eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] Pede did not end normally\n";
	$remark = "pede failed";
	$okStatus = "FAIL";
    }
    if ($pedeLogErr eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] Problems in running Pede:\n";
	print $pedeLogErrStr;
	$remark = "pede error";
	$okStatus = "FAIL";
    }
    if ($pedeLogWrn eq 1) {
# AP 07.09.2009 - Reports Pede Warnings (but do _not_ set job status to FAIL)
	print "@JOBDIR[$i] @JOBID[$i] Warnings in running Pede:\n";
	print $pedeLogWrnStr;
	$remark = "pede warnings";
    }
    if ($endofjob ne 1) {
	print "@JOBDIR[$i] @JOBID[$i] Job not ended\n";
	$remark = "job not ended";
	$okStatus = "FAIL";
    }

    # print warning line to stdout
    if ($okStatus ne "OK") {
      print "@JOBDIR[$i] @JOBID[$i]  --------  $okStatus\n";
    }
    
    # update number of events
    @JOBNEVT[$i] = $nEvent;
    @JOBSTATUS[$i] = $disabled.$okStatus;
    # update CPU time
    @JOBRUNTIME[$i] = $cputime;
    # update host
    ##@JOBHOST[$i] = $farmhost;
    @JOBREMARK[$i] = $remark;
  }
}
write_db();
