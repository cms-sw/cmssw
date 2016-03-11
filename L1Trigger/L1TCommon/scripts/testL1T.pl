#! /bin/env perl

use Switch;

$PYTHON_OPT = "--python_filename=l1t_test.py";
$CMSRUN     = "cmsRun l1t_test.py";

$WORK_DIR = "test_l1t";
$MAIN_LOG = "MAIN.log";
$JOB_LOG = "JOB.log";
$NUM_JOBS = 5;
$TIMEOUT = 10*60;

$VERBOSE  = 0;
$DRYRUN   = 0;
$DELETE   = 0;
$FAST     = 0;
$SLOW     = 0;
$RECYCLE  = 0;
$VISUAL   = 0;
$REDO     = 0;
$REEMUL   = 0;
$COMPARE  = 0;
$COMPARE_DIR = "";

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
    print "--delete           delete previous job directory if it exists.\n";
    print "--fast             limit the number of events for an initial quick test.\n";
    print "--slow             increase the number of events for a slower but more thorough test.\n";
    print "--redo             redo only previously failed tests.\n";
    print "--dryrun           don't launch any long jobs, just show what would be done.\n";
    print "--recycle          recycle previous results, re-running evaluation only (for debugging).\n";
    print "--visual           run some quick, visual checks.\n";
    print "--reemul           generate command-line supported reEmul.py script and exit\n";
    print "--compare=<d>      compare ntuples produced here with those in directory <d>\n";
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
# These are simple visual checks...
#
sub visual_sim_pack_unpack {
    $file = "/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root";
    $status = long_command("cmsDriver.py L1TEST $PYTHON_OPT --conditions auto:run2_mc -s DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI -n 5 --era Run2_2016 --filein=$file --mc --no_output --no_exec  --customise=L1Trigger/Configuration/customiseUtils.L1TStage2SimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TStage2DigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput");

    print "INFO: status of cmsDriver call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }

    $status = long_command("$CMSRUN");
    print "INFO: status of cmsRun call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    print "INFO: visual test has finished without abnormal status...\n";
}

sub visual_unpack {
    $file = "/store/data/Commissioning2016/Cosmics/RAW/v1/000/264/573/00000/5A9E5261-BDD1-E511-9102-02163E014378.root";
    $status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 --conditions=auto:run2_data -n 100 --data --filein=$file --no_output --no_exec --customise=L1Trigger/Configuration/customiseUtils.L1TStage2DigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalMenuXML");

    print "INFO: status of cmsDriver call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    $status = long_command("$CMSRUN");
    print "INFO: status of cmsRun call is $status\n";
    if ($status){
	print "ERROR: abnormal status returned: $status\n";
	return;
    }
    print "INFO: visual test has finished without abnormal status...\n";
}



#
# This is a dummy test:
#
sub test_dummy {
    # for process cleanup to work, make system calls to long processes like this:
    if (! $RECYCLE) {
	long_command("sleep 20");
    }
    system "touch SUCCESS";
}

#
# a simple check that unpackers do not crash on recent RAW data
#
sub test_unpackers_dont_crash {
    $file = "/store/data/Commissioning2016/Cosmics/RAW/v1/000/264/573/00000/5A9E5261-BDD1-E511-9102-02163E014378.root";
    $nevt = 500;
    if ($FAST) {$nevt = 10; }
    if ($SLOW) {$nevt = -1; }
    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 --conditions=auto:run2_data -n $nevt --data --filein=$file --no_output  --no_exec >& CMSDRIVER.log");
	print "INFO: status of cmsDriver call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
	$status = long_command("$CMSRUN >& CMSRUN.log");
	print "INFO: status of cmsRun call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
    }
    system "touch SUCCESS";
}

#
# check that pack unpack is unity
#
sub test_pack_unpack_is_unity {
    # this one runs a bit slower so scale number of events:
    $nevt = 50;
    if ($FAST) {$nevt = 5; }
    if ($SLOW) {$nevt = 500; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT --conditions auto:run2_mc -s DIGI,L1,DIGI2RAW,RAW2DIGI -n $nevt --era Run2_2016 --mc --no_output --no_exec --filein=/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root >& CMSDRIVER.log");
# --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs

	print "INFO: status of cmsDriver call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
	$status = long_command("$CMSRUN >& CMSRUN.log");
	print "INFO: status of cmsRun call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
    }
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
    $nevt = 100;
    if ($FAST) {$nevt = 10; }
    if ($SLOW) {$nevt = 1000; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo --conditions=auto:run2_data -n $nevt --data --no_exec --no_output --filein=$file --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput >& CMSDRIVER.log");

	print "INFO: status of cmsDriver call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
	$status = long_command("$CMSRUN >& CMSRUN.log");
	print "INFO: status of cmsRun call is $status\n";
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
    }

    $SUCCESS = 0;
    open INPUT,"root -b -q -x ../../L1Trigger/L1TCommon/macros/CheckL1Ntuple.C |";
    while (<INPUT>){
	print $_;
	if (/SUCCESS/){	$SUCCESS = 1; }    
    }
    close INPUT;

    if (! $SUCCESS){ 
	print "ERROR:  L1Ntuple did not contain sufficient Calo and Muon candidates for success.\n";
	return;
    }
    
    #print "INFO: parsing the following menu summary:\n";
    #system "grep 'L1T menu Name' -A 250 CMSRUN.log";

    @TRIGGERS = ("L1_SingleMu5","L1_SingleEG5","L1_SingleJet52");
    foreach $trig (@TRIGGERS) {	
	open INPUT,"grep 'L1T menu Name' -A 250 CMSRUN.log | grep $trig |";
	$FIRED = 0;
	while (<INPUT>){
	    #chomp; print "LINE:  $_\n";
	    /$trig\W+(\w+)/;
	    print "INFO:  $trig fired $1 times\n";
	    if ($1 > 0){ $FIRED = 1; }
	}
	if (! $FIRED){
	    print "ERROR:  $trig did not fire.\n";
	    return;
	}
    }
    system "touch SUCCESS";

}

sub test_mc_prod {
    $nevt = 50;
    if ($FAST) {$nevt = 5; }
    if ($SLOW) {$nevt = 500; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST --conditions auto:run2_mc -s DIGI,L1 --datatier GEN-SIM-RAW -n $nevt --era Run2_2016 --mc --no_output --no_exec --filein=/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMUNoEventTree >& CMSDRIVER.log");
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
    if ($RECYCLE){
	if (! -e $JOBDIR){
	    print "ERROR:  --recycle specified but $JOBDIR does not exist!\n";
	    return;
	}
	chdir $JOBDIR;
	if (-e "SUCCESS"){ system "rm SUCCESS"; }
    } else {
	system "mkdir $JOBDIR";
	chdir $JOBDIR;
    }
    open STDOUT,">",$JOB_LOG or die $!;
    open STDERR,">",$JOB_LOG or die $!;
    print "INFO: job $ijob starting...\n";
    my $start_time = time();
    switch ($ijob){
#	case 0 {test_dummy; }
	case 0 {test_reemul;}
	case 1 {test_mc_prod; }
	#case 2 {test_unpackers_dont_crash; }
	case 2 {test_dummy; }
	case 3 {test_pack_unpack_is_unity; }
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
            elsif ($arg =~ /--verbose/)   { $VERBOSE   = 1;          }
            elsif ($arg =~ /--dryrun/)    { $DRYRUN    = 1;          }
            elsif ($arg =~ /--delete/)    { $DELETE    = 1;          }
            elsif ($arg =~ /--fast/)      { $FAST      = 1;          }
            elsif ($arg =~ /--slow/)      { $SLOW      = 1;          }
            elsif ($arg =~ /--recycle/)   { $RECYCLE   = 1;          }
            elsif ($arg =~ /--visual/)    { $VISUAL    = 1;          }
            elsif ($arg =~ /--redo/)      { $REDO      = 1;          }
            elsif ($arg =~ /--reemul/)    { $REEMUL    = 1;          }
            elsif ($arg =~ /--compare=(\S+)/) { $COMPARE = 1; $COMPARE_DIR=$1;}
	    else {print "ERROR: unrecognized argument: $arg\n"; usage(); }
        } else {
            push @args, $arg;
        }
    }    
    if ($#args != -1){ usage(); }

    if ($FAST && $SLOW){ usage(); }
    if ($FAST){ $WORK_DIR = "${WORK_DIR}_fast"; $TIMEOUT = 5*60;}
    if ($SLOW){ $WORK_DIR = "${WORK_DIR}_slow"; $TIMEOUT = 30*60;}
			    
    print "INFO: Welcome the L1T offline software integration testing!\n";

    if ($REDO){
	print "INFO:  --redo option not yet implemented.";
	exit(0);
    }
    if ($REEMUL){
	$nevt = 10;
	$file = "/store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/4A874FB5-585D-E511-A3D8-02163E0143B5.root";
	$status = long_command("cmsDriver.py L1TEST --python_filename=reEmul.py -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo --conditions=auto:run2_data -n $nevt --data --no_exec --no_output --filein=$file --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs >& CMSDRIVER.log");
	if ($status){
	    print "ERROR: abnormal status returned: $status\n";
	    return;
	}
	print "INFO: adding command line option support\n";
        system "cat L1Trigger/L1TCommon/scripts/optionsL1T.py >> reEmul.py";
	exit(0);	
    }
    if ($COMPARE){
	print "INFO: Comparing results with those in $COMPARE_DIR\n";
	$ours = "$WORK_DIR/test_0/L1Ntuple.root";
	$theirs = "$COMPARE_DIR/$WORK_DIR/test_0/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x 'L1Trigger/L1TCommon/macros/NtupleDiff.C(\"reemul\",\"$ours\",\"$theirs\")'");

	$ours = "$WORK_DIR/test_1/L1Ntuple.root";
	$theirs = "$COMPARE_DIR/$WORK_DIR/test_1/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x 'L1Trigger/L1TCommon/macros/NtupleDiff.C(\"mc\",\"$ours\",\"$theirs\")'");

	# this is a hack until L1T uGT output goes into L1TNtuple:
	system "grep \"L1T menu Name\" -A 109 $WORK_DIR/test_0/CMSRUN.log > menu_a.txt";
	system "grep \"L1T menu Name\" -A 109 $COMPARE_DIR/$WORK_DIR/test_0/CMSRUN.log > menu_b.txt";
	print "INFO:  diff of menu summary follows:\n";
	system "diff menu_a.txt menu_b.txt\n";
	exit(0);
    }



    if ($VISUAL){
	print "INFO: visual mode was specified... note that successful results do not green-light a commit.\n";
	visual_unpack();
	visual_sim_pack_unpack();
	exit(0);
    }
    
    open(VOMS, "voms-proxy-info |");
    $VOMS_SUCCESS = 0;
    while (<VOMS>){
	if (/timeleft/){$VOMS_SUCCESS=1;}
    }
    
    if ($VOMS_SUCCESS){
	print "INFO:  check of voms-proxy=info succeeded.\n";
    } 

    if (! $VOMS_SUCCESS) {
	print "ERROR: you must call voms-proxy-init first, in order to access remote files for tests!\n";
	return;
    }


    if ($FAST){
	print "INFO: fast mode was specified... note that successful results will not green-light a commit.\n";
    }

    if (! $RECYCLE){
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
    } else {
	if (! -e $WORK_DIR){
	    print "ERROR: --recycle specified but $WORK_DIR does not exist yet....\n";
	    return;
	}
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
	if ($SLOW){ print "INFO:  Slow mode specified... better make it two!\n"; }
	print "INFO: You can view progress at:\n";
	print "$LOG\n";
        # for some reason, this not true:
	# print "INFO: This process will now begin tailing the LOG file, but you do not need to stick around:\n";	
        #print "INFO: you may safely exit this process, without disturbing the running jobs, using Ctrl^C.\n";
	#my $pid = fork();
	#die "$0: fork: $!" unless defined $pid;
	#if ($pid) {
	#    exit(0);
	#}
	exec("tail -f $LOG");
	#exit(0);
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

    #while($NRUNNING > 0){
#	for ($i=0; $i <= $#PIDS; $i++){
#	
#	} 
 #   }

    while($#PIDS > -1){
	my $pid = wait;
	print "INFO: job with $pid has finished.\n";
	my $index = 0;
	$index++ until $PIDS[$index] == $pid;
	splice(@PIDS, $index, 1);
    }
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
	if ($FAST || $DRYRUN || $RECYCLE){
	    print "STATUS:  results not sufficient for greenlight to commit due to --fast, --dryrun, or --recycle option\n";
	} 
	else {
	    print "STATUS:  you have GREEN LIGHT to commit your L1T code!\n";
	    system "touch GREEN_LIGHT";
	}
    }
}
