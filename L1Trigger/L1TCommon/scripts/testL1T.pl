#! /bin/env perl

use Switch;

$WORK_DIR = "./test_l1t";
$MAIN_LOG = "MAIN.log";
$JOB_LOG = "JOB.log";
$NUM_JOBS = 3;
$TIMEOUT = 10*60;
$NUM_EVENTS = 100;

$VERBOSE  = 0;
$DRYRUN   = 0;
$DELETE   = 0;
$FAST     = 0;
sub main;
main @ARGV;

sub usage() {
    print "usage:  testL1T.pl [opt]\n";
    print "\n";
    print "Integration test for L1T.\n";
    print "\n";
    print "Possible options:\n";
    print "--help             display this message.\n";
    print "--verbose          output lots of information.\n";
    print "--dryrun           don't launch any long jobs, just show what would be done.\n";
    print "--delete           delete previous job directory.\n";
    print "--fast             limit the number of events for an initial quick test.\n";
    exit 0;
}

$CHILD_PID = 0;
sub long_command {
    my $cmd = shift;
    if ($DRYRUN){
	print "INFO: due to --dryrun, not running command:  $cmd\n";
	return 0;
    }
    print "INFO: running command:  $cmd\n";
    $CHILD_PID = fork() or exec($cmd);
    waitpid( -1, WNOHANG );    
    $CHILD_PID = 0;
    return $?;
}

#
# This is a dummy test:
#
sub test_dummy {
    # for process cleanup to work, make system calls to long processes like this:
    long_command("sleep 20");
    system "touch SUCCESS";
}

#
# Test the re-emulation sequence:
#
# - checks cmsDriver.py command for Stage-2 full re-emulation from RAW, saving re-emulation output to ntuple
# - checks cmsRun on resultant config
# - checks ntuple for non-zero number of jets, taus, e-gammas, and muons.
#
sub test_reemul {
#    $file = "/store/data/Run2015D/DoubleEG/RAW-RECO/ZElectron-PromptReco-v4/000/260/627/00000/12455212-1E85-E511-8913-02163E014472.root";
    $file = "/store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/4A874FB5-585D-E511-A3D8-02163E0143B5.root";

    $status = long_command("cmsDriver.py L1TEST -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo --conditions=auto:run2_data -n $NUM_EVENTS --data --no_exec --no_output --filein=$file --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs >& CMSDRIVER.log");

    print "INFO: status of cmsDriver call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    $status = long_command("cmsRun L1TEST_RAW2DIGI.py >& CMSRUN.log");
    print "INFO: status of cmsRun call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }

    open INPUT,"root -b -q -x ../../L1Trigger/L1TCommon/macros/CheckL1Ntuple.C |";
    while (<INPUT>){
	print $_;
	if (/SUCCESS/){	    
	    system "touch SUCCESS";
	}
    }
    # not enough muons in fast mode... for now just declare success if we make it this far:
    #if ($FAST){
	#system "touch SUCCESS";
    #}

}

sub test_mc_prod {
    # this one runs a bit slower so scale number of events:
    my $nevt = $NUM_EVENTS * 0.5;
    if ($nevt < 5) { $nevt = 5; }

    $status = long_command("cmsDriver.py L1TEST --conditions auto:run2_mc -s DIGI,L1 --datatier GEN-SIM-RAW -n $NUM_EVENTS --era Run2_2016 --mc --no_output --no_exec --filein=/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMUNoEventTree >& CMSDRIVER.log");
# --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs

    print "INFO: status of cmsDriver call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    $status = long_command("cmsRun L1TEST_DIGI_L1.py >& CMSRUN.log");
    print "INFO: status of cmsRun call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    open INPUT,"root -b -q -x ../../L1Trigger/L1TCommon/macros/CheckL1Ntuple.C |";
    while (<INPUT>){
	print $_;
	if (/SUCCESS/){	    
	    system "touch SUCCESS";
	}
    }
}




sub run_job {
    my $ijob = shift;
    $SIG{HUP} = sub { 
	print "ERROR: job $ijob received HUP.. exiting\n"; 
	if ($CHILD_PID) { kill HUP => $CHILD_PID; }
	exit 0; 
    };
    $JOBDIR = "test_$ijob";
    system "mkdir $JOBDIR";
    chdir $JOBDIR;
    open STDOUT,">",$JOB_LOG or die $!;
    open STDERR,">",$JOB_LOG or die $!;
    print "INFO: job $ijob starting...\n";
    my $start_time = time();
    switch ($ijob){
	case 0 {test_dummy; }
#	case 0 {test_reemul;}
#	case 1 {test_mc_prod; }
	else   {test_dummy; }
    }
    my $job_time = time() - $start_time;
    print "INFO: job $ijob ending after $job_time seconds...\n";
    exit;
}


sub main {
    my @args = ();
    # parse the command line arguments:
    my $arg;
    while($arg = shift){
        if ($arg =~ /^--/){
            if ($arg =~ /--help/)      { usage();                 }
            if ($arg =~ /--verbose/)   { $VERBOSE   = 1;          }
            if ($arg =~ /--dryrun/)    { $DRYRUN    = 1;          }
            if ($arg =~ /--delete/)    { $DELETE    = 1;          }
            if ($arg =~ /--fast/)      { $FAST      = 1;          }
        } else {
            push @args, $arg;
        }
    }    
    if ($#args != -1){ usage(); }

    print "INFO: Welcome the L1T offline software integration testing!\n";

    if ($FAST){
	print "INFO: fast mode was specified... note that successful results will not green-light a commit.\n";
	$NUM_EVENTS = 5;
    }

    if (-e $WORK_DIR){
	if (!$DELETE){
	    print "ERROR: cowardly refusing to overwrite existing test directory: $WORK_DIR\n";
	    print "ERROR: (move or delete it yourself)\n";
	    return;
	} else {
	    system "rm -fr $WORK_DIR";
	    if (-e $WORK_DIR){
		print "ERROR: could not delete $WORK_DIR\n";
		return;
	    }
	}
    }
    system "mkdir $WORK_DIR";
    if (! -e $WORK_DIR){
	print "ERROR: could not create $WORK_DIR\n";
	return;	    
    }

    $start_time = time();

    chdir $WORK_DIR;
    $PWD = `pwd`;
    chomp $PWD;
    $LOG = "$PWD/$MAIN_LOG";

    my $pid = fork();
    die "$0: fork: $!" unless defined $pid;

    if ($pid) {
	print "INFO: Coffee Time!!!\n";
	print "INFO: You can view progress at:\n";
	print "$LOG\n";
	print "INFO: This process will now begin tailing the LOG file, but you do not need to stick around:\n";
	print "INFO: you may safely exit this process, without disturbing the running jobs, using Ctrl^C.\n";
	exec("tail -f $LOG");
    }
    
    open STDOUT,">",$LOG or die $!;
    open STDERR,">",$LOG or die $!;

    print "INFO: launching $NUM_JOBS test jobs...\n";
    @PIDS   = ();
    @STATUS = ();
    $NRUNNING = 0;
    for ($ijob = 0; $ijob < $NUM_JOBS; $ijob++){
	print "INFO: launching job number $ijob\n";	
	my $pid = fork();
	die "$0: fork: $!" unless defined $pid;

	if ($pid) {
	    print "INFO: child process $pid launched...\n";
	    push @PIDS, $pid;
	    push @STATUS, 0;
	    $NRUNNING++;
	} else {
	    run_job($ijob);
	}	
    }
 

    #if (eval { }){
#	print "SUCCESS...\n";"
#    }


    $SIG{ALRM} = sub {
	print "ERROR: timeout waiting for child processes...\n"; 
	foreach $pid (@PIDS) {
	    print "ERROR: killing child process $pid\n";
	    kill HUP => $pid;
	}
	print "ERROR: exiting before all jobs complete..\n";
	summary();
	exit 0;
    };
   
    alarm $TIMEOUT;

    

    while($NRUNNING > 0){
	for ($i=0; $i <= $#PIDS; $i++){

	} 

        #my $pid = wait;
	#print "INFO: job with $pid has finished.\n";
	#my $index = 0;
	#$index++ until $PIDS[$index] == $pid;
	#splice(@PIDS, $index, 1);
    #}
    alarm 0;
    print "INFO: all jobs have finished.\n";	

    $job_time = time() - $start_time;
    print "INFO: testL1T took $job_time seconds to complete.\n";
    summary();



}
    
sub summary {
    $FAIL = 0;
    for ($ijob = 0; $ijob < $NUM_JOBS; $ijob++){
	$SUCCESSFILE = "test_$ijob/SUCCESS";
	if (-e $SUCCESSFILE) { print "STATUS:  tests job $ijob result was SUCCESS\n"; }
	else                 { 
	    print "STATUS:  tests job $ijob result was FAIL\n";
	    $FAIL = 1;
	}
    }
    if ($FAIL) { print "STATUS:  testL1T overall status:  FAIL\n"; }
    else       { 
	print "STATUS:  testL1T overall status:  SUCCESS\n"; 
	if ($FAST || $DRYRUN){
	    print "STATUS:  results not sufficient for greenlight to commit due to --fast or --dryrun option\n";
	} 
	else {
	    print "STATUS:  you have GREEN LIGHT to commit your L1T code!\n";
	    system "touch GREEN_LIGHT";
	}
    }
}
