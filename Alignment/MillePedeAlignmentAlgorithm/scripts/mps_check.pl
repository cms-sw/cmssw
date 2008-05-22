#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     09-Jul-2007
#     A. Parenti, DESY Hamburg    24-Apr-2008
#     $Revision: 1.3 $
#     $Date: 2008/04/20 18:32:10 $
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
# loop over FETCH jobs
for ($i=0; $i<@JOBID; ++$i) {
  $batchSuccess = 0;
  $batchExited = 0;
  $finished = 0;
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
  $exceptionCaught = 0;
  $timeout = 0;
  $cfgerr = 0;
  $emptyDatErr = 0;

  $remark = "";

  if (@JOBSTATUS[$i] eq "FETCH") {

    # open the STDOUT file
    $stdOut = "jobData/@JOBDIR[$i]/STDOUT";
    open STDFILE,"$stdOut";
    # scan records in input file
    $nEvent = -1;
    while ($line = <STDFILE>) {
      if (($line =~ m/CERN report: Job Killed/) eq 1) { $killed = 1;}
      if (($line =~ m/Job finished/) eq 1)  { $finished = 1; }
      if (($line =~ m/connection timed out/) eq 1)  { $timeout = 1; }
      if ($line =~ m/This job used .+?(\d+) NCU seconds/ eq 1) {
	$ncuFactor = 3;
	$cputime = $1 / $ncuFactor;
	# print "Set cpu to $cputime\n";
      }
      if (($line =~ m/ConfigFileReadError/) eq 1)  { $cfgerr = 1; }

    }
    close STDFILE;

    $eazeLog = "jobData/@JOBDIR[$i]/cmsRun.out";

    # open the input file
    open INFILE,"$eazeLog";
    # scan records in input file
    $nEvent = 0;
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
      if (($line =~ m/\<EventsRead\> *(\d+)\<\/EventsRead\>/) eq 1) {
	$nEvent = $nEvent + $1;
      }
    }
    close INFILE;

    # if there is an alignment.log[.gz] file, check it as well
    $eazeLog = "jobData/@JOBDIR[$i]/alignment.log";
    $logZipped = "no";
    if (-r $eazeLog.".gz") {
      system "gunzip ".$eazeLog.".gz";
      $logZipped = "true";
    }
    # open the input file
    open INFILE,"$eazeLog";
    # scan records in input file
    $nEvent = 0;
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
      if (($line =~ m/Exception caught in cmsRun/) eq 1) { $exceptionCaught = 1;}
      if (($line =~ m/\<EventsRead\> *(\d+)\<\/EventsRead\>/) eq 1) {
	$nEvent = $nEvent + $1;
      }
    }
    close INFILE;
    if ($logZipped eq "true") {
      system "gzip $eazeLog";
    }

    # for mille jobs checks that milleBinary file is not empty
    if ( @JOBDIR[$i] ne "jobm" && !($mOutSize>0) ) {
      $milleOut = sprintf("$mssDir/milleBinary%03d.dat",$i+1);
      $mOutSize = `nsls -l $milleOut | awk '{print \$5}'`;
      $emptyDatErr = 1;
    }

    # additional checks for merging job
    if (@JOBDIR[$i] eq "jobm") {
        # if there is an alignment_merge.log[.gz] file, check it as well
	$eazeLog = "jobData/@JOBDIR[$i]/alignment_merge.log";
        $logZipped = "no";
        if (-r $eazeLog.".gz") {
          system "gunzip ".$eazeLog.".gz";
          $logZipped = "true";
        }
	if (-r $eazeLog) {
	    # open the input file
	    open INFILE,"$eazeLog";
	    # scan records in input file
	    $nEvent = 0;
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
		if (($line =~ m/\<EventsRead\> *(\d+)\<\/EventsRead\>/) eq 1) {
		    $nEvent = $nEvent + $1;
		}
	    }
	    close INFILE;
	} else {
	    print "mps_check.pl cannot find $eazeLog to test";
	}
	if ($logZipped eq "true") {
	    system "gzip $eazeLog";
	}

	# if there is a pede.dump file, check it as well
	$eazeLog = "jobData/@JOBDIR[$i]/pede.dump";
	if (-r $eazeLog) {
	    # open the input file
	    open INFILE,"$eazeLog";
	    # scan records in input file
	    $pedeAbend = 1;
	    while ($line = <INFILE>) {
		# check if pede has reached its normal end
		if (($line =~ m/Millepede II ending/) eq 1) { $pedeAbend = 0;}
	    }
	}

    }

    $farmhost = " ";

    $okStatus = "OK";
    unless ($eofile eq 1) {
      print "@JOBDIR[$i] @JOBID[$i] did not reach end of file\n";
      $okStatus = "ABEND";
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
      $okStatus = "FAIL"; 
    }

    if ($pedeAbend eq 1) {
	print "@JOBDIR[$i] @JOBID[$i] Pede did not end normally\n";
	$remark = "pede failed";
	$okStatus = "FAIL";
    }

    # print warning line to stdout
    if ($okStatus ne "OK") {
      print "@JOBDIR[$i] @JOBID[$i]  --------  $okStatus\n";
    }
    
    # update number of events
    @JOBNEVT[$i] = $nEvent;
    @JOBSTATUS[$i] = $okStatus;
    # update CPU time
    @JOBRUNTIME[$i] = $cputime;
    # update host
    ##@JOBHOST[$i] = $farmhost;
    @JOBREMARK[$i] = $remark;
  }
}
write_db();
