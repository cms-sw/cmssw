#! /bin/env perl

use Switch;
use POSIX "waitpid";
use File::Basename;

$PYTHON_OPT = "--python_filename=l1t_test.py";
$CMSRUN     = "cmsRun l1t_test.py";

$WORK_DIR = "test_l1t";
$MAIN_LOG = "MAIN.log";
$JOB_LOG = "JOB.log";
$DIE_FILE = "DIE";
$NUM_JOBS = 8;
$TIMEOUT = 10*60;
#$GEN_SIM_FILE = "/store/relval/CMSSW_7_6_0/RelValProdTTbar/GEN-SIM/76X_mcRun1_realistic_v10-v1/00000/08F44119-9C7F-E511-85C3-0025905B85A2.root";
#$GEN_SIM_FILE = "file:/afs/cern.ch/work/m/mulhearn/data/gen-sim/GEN-SIM-08F44119-9C7F-E511-85C3-0025905B85A2.root";
#$GEN_SIM_FILE = "root://cms-xrd-global.cern.ch///store/relval/CMSSW_7_6_0/RelValProdTTbar/GEN-SIM/76X_mcRun1_realistic_v10-v1/00000/08F44119-9C7F-E511-85C3-0025905B85A2.root";
$GEN_SIM_FILE = "root://eoscms//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/unit-test/input//store/relval/CMSSW_7_6_0/RelValProdTTbar/GEN-SIM/76X_mcRun1_realistic_v10-v1/00000/08F44119-9C7F-E511-85C3-0025905B85A2.root";
#$DATA_2016_FILE = "root://eoscms//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/unit-test/input//store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root";
$DATA_2016_FILE = "/store/data/Run2016H/HLTPhysics3/RAW/v1/000/281/602/00000/0E71643D-BF82-E611-8593-02163E011B19.root";
#$DATA_2017_FILE = "/store/express/Run2017C/ExpressPhysics/FEVT/Express-v3/000/301/142/00000/0008C605-0881-E711-84A2-02163E019C07.root";
#$DATA_2017_FILE = "/store/data/Run2017C/HLTPhysics/RAW/v1/000/301/142/00000/00682207-3181-E711-8706-02163E01445A.root";
#$DATA_2017_FILE = "/store/data/Run2017G/HLTPhysics/RAW/v1/000/306/826/00000/3EFDFCB7-A2CE-E711-80E1-02163E019BEA.root";
#$DATA_2017_FILE = "root://cms-xrd-global.cern.ch///store/data/Run2017H/SingleMuon/RAW/v1/000/307/082/00000/FE78107C-90D2-E711-84AB-02163E01A586.root";
#$DATA_2018_FILE = "/store/data/Run2018A/HLTPhysics/RAW/v1/000/316/994/00000/CEFB7B46-3B62-E811-BA1F-FA163E4C0F07.root";
#$DATA_2017_FILE = "root://cms-xrd-global.cern.ch///store/data/Run2017C/JetHT/RAW/v1/000/301/914/00000/C4B8EA5E-988A-E711-B38B-02163E01399E.root";
$DATA_2017_FILE = "root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/unit-test/input//store/data/Run2017C/JetHT/RAW/v1/000/301/914/00000/C4B8EA5E-988A-E711-B38B-02163E01399E.root";
#$DATA_2017_FILE = "root://cms-xrd-global.cern.ch///store/express/Run2017C/ExpressPhysics/FEVT/Express-v2/000/300/226/00000/0218D0FB-2376-E711-B5B1-02163E014257.root";
#$DATA_2017_FILE = "root://cms-xrd-global.cern.ch///store/data/Run2017D/HLTPhysics9/RAW/v1/000/302/676/00000/7A9DB54D-6D98-E711-B9AE-02163E013478.root";
#$DATA_2018_FILE = "root://cms-xrd-global.cern.ch///store/data/Run2018A/SingleMuon/RAW/v1/000/316/218/00000/3CA8127E-4E57-E811-887E-FA163EEA1DC3.root";
#$DATA_2018_FILE = "root://cms-xrd-global.cern.ch///store/express/Run2017F/ExpressPhysics/FEVT/Express-v1/000/305/636/00002/581DD5E7-10BA-E711-8E38-02163E011B94.root";
#$DATA_2018_FILE = "/store/express/Run2018C/ExpressPhysics/FEVT/Express-v1/000/319/992/00000/041AB4A0-708C-E811-921F-02163E0153F7.root";
#$DATA_2018_FILE = "/store/express/Run2018C/ExpressPhysics/FEVT/Express-v1/000/319/992/00000/041AB4A0-708C-E811-921F-02163E0153F7.root";
#$DATA_2018_FILE = "root://cms-xrd-global.cern.ch///store/data/Run2018A/JetHT/RAW/v1/000/316/995/00000/FEEAE7BF-CA64-E811-8959-FA163E35C0CD.root";
#$DATA_2018_FILE = "root://eoscms//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/unit-test/input//data/Run2018A/JetHT/RAW/v1/000/316/995/00000/FEEAE7BF-CA64-E811-8959-FA163E35C0CD.root";
$DATA_2018_FILE = "root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/unit-test/input/store/data/Run2018B/JetHT/RAW/v1/000/316/995/00000/FEEAE7BF-CA64-E811-8959-FA163E35C0CD.root";
#cmsDriver.py $PYTHON_OPT -s RAW2DIGI --era=Run2_2018 --data --conditions=101X_dataRun2_Prompt_v9 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleAODRAWEMU --customise=L1Trigger/Configuration/customiseSettings.L1TSettingsToCaloParams_2018_v1_2 --filein=/store/express/Run2017F/ExpressPhysics/FEVT/Express-v1/000/305/636/00002/581DD5E7-10BA-E711-8E38-02163E011B94.root

$VERBOSE  = 0;
$KILL     = 0;
$DRYRUN   = 0;
$DELETE   = 0;
$FAST     = 0;
$SLOW     = 0;
$RECYCLE  = 0;
$VISUAL   = 0;
$REDO     = 0;
$REEMUL   = 0;
$COMPARE  = 0;
$COMPARE2  = 0;
$COMPARE_DIR = "";
$COMPARE_DIRS = "";
$SINGLE  = 0;
$SINGLE_JOB  = 0;

$COND_MC   = "--conditions=auto:run2_mc";
#$COND_MC   = "--conditions=90X_mcRun2_asymptotic_v0";
#$COND_MC   = "--conditions=90X_upgrade2017_realistic_v6_C1";
#$COND_DATA_2015 = "--conditions=auto:run2_data";
#$COND_DATA_2016 = "--conditions=auto:run2_data";
#$COND_DATA_2017 = "--conditions=auto:run2_data";
#$COND_DATA_2018 = "--conditions=auto:run2_data";
$COND_DATA_2015 = "--conditions=101X_dataRun2_Prompt_v9";
$COND_DATA_2016 = "--conditions=101X_dataRun2_Prompt_v9";
$COND_DATA_2017 = "--conditions=101X_dataRun2_Prompt_v9";
$COND_DATA_2018 = "--conditions=101X_dataRun2_Prompt_v9";
#$COND_DATA_2017 = "--conditions=92X_dataRun2_Prompt_v8";
#$COND_MC   = "--conditions=80X_mcRun2_asymptotic_v17";  # = run2_mc in CMSSW_8_0_19 
#$COND_DATA_2015 = "--conditions=80X_dataRun2_v18";      # = run2_data in CMSSW_8_0_19 
#$COND_DATA_2016 = "--conditions=80X_dataRun2_Prompt_v8";

sub main;
main @ARGV;

sub usage() {
    print "usage: testL1T.pl [opt]\n";
    print "\n";
    print "Integration test for L1T.\n";
    print "\n";
    print "Possible options:\n";
    print "--help             display this message.\n";
    print "--kill             cleanly kill a running instance of testL1T.pl.\n";
    print "--verbose          output lots of information.\n";
    print "--delete           delete previous job directory if it exists.\n";
    print "--fast             limit the number of events for an initial quick test.\n";
    print "--slow             increase the number of events for a slower but more thorough test.\n";
    print "--dryrun           don't launch any long jobs, just show what would be done.\n";
    print "--recycle          recycle previous results, re-running evaluation only (for debugging).\n";
    print "--visual           run some quick, visual checks.\n";
    print "--reemul           generate command-line supported reEmul.py script and exit\n";
    print "--compare=<d>      compare ntuples produced here with those in directory <d>\n";
    print "--compare2=<d1:d2> compare ntuples produced in directory <d1> with those in directory <d2>\n";
    print "--single=<d>       only run single job <d>\n";
    print "--redo             redo any failed jobs\n";
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
    $status = long_command("cmsDriver.py L1TEST $PYTHON_OPT $COND_MC -s DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI -n 5 --era Run2_2016 --filein=$GEN_SIM_FILE --mc --no_output --no_exec  --customise=L1Trigger/Configuration/customiseUtils.L1TStage2SimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TStage2DigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput");

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
    $status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 $COND_DATA_2015 -n 100 --data --filein=$file --no_output --no_exec --customise=L1Trigger/Configuration/customiseUtils.L1TStage2DigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalMenuXML");

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
    #$file = "/store/data/Commissioning2016/Cosmics/RAW/v1/000/264/573/00000/5A9E5261-BDD1-E511-9102-02163E014378.root";
    #$file = "/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root";
    $nevt = 200;
    if ($FAST) {$nevt = 10; }
    if ($SLOW) {$nevt = -1; }
    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 $COND_DATA_2015 -n $nevt --data --filein=$DATA_2016_FILE --no_output  --no_exec >& CMSDRIVER.log");
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
# Test unpack non-zero payloads from 2016 Data
#
sub test_unpack_2016_data {
    # this one runs a bit slower so scale number of events:
    $nevt = 200;
    if ($FAST) {$nevt = 100; }
    if ($SLOW) {$nevt = 500; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT $COND_DATA_2016 -s RAW2DIGI -n $nevt --era=Run2_2016 --data --filein=$DATA_2016_FILE --no_output --no_exec --customise=L1Trigger/Configuration/customiseUtils.L1TStage2DigisSummary --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleRAW >& CMSDRIVER.log");

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
    open INPUT,"root -b -q -x '../../L1Trigger/L1TCommon/macros/CheckL1Ntuple.C(\"L1Ntuple.root\",\"l1UpgradeTree/L1UpgradeTree\")' |";
    while (<INPUT>){
	print $_;
	if (/SUCCESS/){	    
	    system "touch SUCCESS";
	}
    }
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
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT $COND_MC -s DIGI,L1,DIGI2RAW,RAW2DIGI -n $nevt --era Run2_2016 --mc --no_output --no_exec --filein=$GEN_SIM_FILE --customise=L1Trigger/Configuration/customiseUtils.L1TStage2ComparisonRAWvsEMU --customise=L1Trigger/Configuration/customiseUtils.L1TGtStage2ComparisonRAWvsEMU >& CMSDRIVER.log");
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
    $count = 0;
    open (INPUT, "grep \"SUMMARY:  L1T Comparison\" CMSRUN.log |");
    while (<INPUT>){
	if (/SUCCESS/) {$count++;}
    }
    if ($count == 2) {system "touch SUCCESS"; }
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
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW2015 --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU $COND_DATA_2015 -n $nevt --data --no_exec --no_output --filein=$file --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput >& CMSDRIVER.log");

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

sub test_reemul_2016 {
#    $file = "/store/data/Run2015D/DoubleEG/RAW-RECO/ZElectron-PromptReco-v4/000/260/627/00000/12455212-1E85-E511-8913-02163E014472.root";
    #$file = "/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root";
    #$file = "/store/data/Run2016B/MuonEG/RAW-RECO/TopMuEG-PromptReco-v2/000/273/450/00000/8EF765B2-531C-E611-86D5-02163E012060.root";
    #$file = "/store/data/Run2016B/ZeroBias8/RAW/v1/000/272/021/00000/D821E02F-BA0D-E611-9DE0-02163E013885.root";
    $nevt = 100;
    if ($FAST) {$nevt = 200; }
    if ($SLOW) {$nevt = 1000; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py $PYTHON_OPT -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo $COND_DATA_2015 -n $nevt --data --no_exec --no_output --filein=$DATA_2016_FILE --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalMenuXML >& CMSDRIVER.log");

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

    @TRIGGERS = ("L1_SingleMu3","L1_SingleEG5","L1_SingleJet16");
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

sub test_reemul_2017 {
    $nevt = 100;
    if ($FAST) {$nevt = 100; }
    if ($SLOW) {$nevt = 1000; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2017 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo $COND_DATA_2017 -n $nevt --data --no_exec --no_output --filein=$DATA_2017_FILE --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput >& CMSDRIVER.log");

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

    @TRIGGERS = ("L1_SingleMu3","L1_SingleEG5","L1_SingleJet16");
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

sub test_reemul_2018 {
    #$file = "/store/express/Run2017C/ExpressPhysics/FEVT/Express-v3/000/301/142/00000/0008C605-0881-E711-84A2-02163E019C07.root"
    $nevt = 100;
    if ($FAST) {$nevt = 100; }
    if ($SLOW) {$nevt = 1000; }

    if (! $RECYCLE){
	$status = long_command("cmsDriver.py L1TEST $PYTHON_OPT -s RAW2DIGI --era=Run2_2018 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo $COND_DATA_2018 -n $nevt --data --no_exec --no_output --filein=$DATA_2018_FILE --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput >& CMSDRIVER.log");

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

    @TRIGGERS = ("L1_SingleMu3","L1_SingleEG50","L1_SingleJet35");
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
	$status = long_command("cmsDriver.py L1TEST $COND_MC -s DIGI,L1 --datatier GEN-SIM-RAW -n $nevt --era Run2_2016 --mc --no_output --no_exec --filein=$GEN_SIM_FILE --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMUNoEventTree  --customise=L1Trigger/Configuration/customiseUtils.L1TGlobalSimDigisSummary --customise=L1Trigger/Configuration/customiseUtils.L1TAddInfoOutput >& CMSDRIVER.log");
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
	#case 0 {test_dummy; }
	case 0 {test_reemul;}
	case 1 {test_mc_prod; }
	case 2 {test_unpack_2016_data; }
	case 3 {test_pack_unpack_is_unity; }
	case 4 {test_unpackers_dont_crash; }
	case 5 {test_reemul_2016;}
	case 6 {test_reemul_2017;}
	case 7 {test_reemul_2018;}
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
            elsif ($arg =~ /--kill/)      { $KILL      = 1;          }
            elsif ($arg =~ /--dryrun/)    { $DRYRUN    = 1;          }
            elsif ($arg =~ /--delete/)    { $DELETE    = 1;          }
            elsif ($arg =~ /--fast/)      { $FAST      = 1;          }
            elsif ($arg =~ /--slow/)      { $SLOW      = 1;          }
            elsif ($arg =~ /--recycle/)   { $RECYCLE   = 1;          }
            elsif ($arg =~ /--visual/)    { $VISUAL    = 1;          }
            elsif ($arg =~ /--reemul/)    { $REEMUL    = 1;          }
            elsif ($arg =~ /--compare=(\S+)/) { $COMPARE = 1; $COMPARE_DIR=$1;}
            elsif ($arg =~ /--compare2=(\S+)/) { $COMPARE2 = 1; $COMPARE_DIRS=$1;}
            elsif ($arg =~ /--single=(\S+)/) { $SINGLE = 1; $SINGLE_JOB=$1;}
            elsif ($arg =~ /--redo/)      { $REDO      = 1;          }
	    else {print "ERROR: unrecognized argument: $arg\n"; usage(); }
        } else {
            push @args, $arg;
        }
    }    
    if ($#args != -1){ usage(); }

    if ($FAST && $SLOW){ usage(); }
    if ($FAST){ $WORK_DIR = "${WORK_DIR}_fast"; $TIMEOUT = 5*60;}
    if ($SLOW){ $WORK_DIR = "${WORK_DIR}_slow"; $TIMEOUT = 30*60;}


    if ($KILL) {
	print "INFO: killing testL1T.pl instance running at $WORK_DIR\n";
	system "touch $WORK_DIR/$DIE_FILE";
	system "sleep 3";
	if (-e "$WORK_DIR/$DIE_FILE"){
	    print "ERROR: kill was not successfull.. perhaps that instance already finished?\n";
	    system "rm $WORK_DIR/$DIE_FILE";
	}
	exit(0);
    }

    print "INFO: Welcome the L1T offline software integration testing!\n";

    if ($SINGLE){
	print "INFO:  will run only single job $SINGLE_JOB\n";
	$WORK_DIR = "${WORK_DIR}_single_${SINGLE_JOB}";    
    }
    if ($REDO){
	print "INFO:  will redo all failed jobs.\n";
    }
			    
    if ($REEMUL){
	$nevt = 10;
	$file = "/store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/4A874FB5-585D-E511-A3D8-02163E0143B5.root";
	$status = long_command("cmsDriver.py L1TEST --python_filename=reEmul.py -s RAW2DIGI --era=Run2_2016 --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW2015 --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.L1NtupleEMU --customise=L1Trigger/Configuration/customiseUtils.L1TTurnOffUnpackStage2GtGmtAndCalo $COND_DATA_2015 -n $nevt --data --no_exec --no_output --filein=$file --geometry=Extended2016,Extended2016Reco --customise=L1Trigger/Configuration/customiseReEmul.L1TEventSetupForHF1x1TPs >& CMSDRIVER.log");
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
        system "sed -n \'/L1T menu Name/,/Final OR Count/p\' $WORK_DIR/test_0/CMSRUN.log > menu_a.txt";
        system "sed -n \'/L1T menu Name/,/Final OR Count/p\' $COMPARE_DIR/$WORK_DIR/test_0/CMSRUN.log > menu_b.txt";
	print "INFO:  diff of menu summary follows:\n";
	system "diff menu_a.txt menu_b.txt\n";
	exit(0);
    }

    if ($COMPARE2){

        if ($FAST) {$nevt = 5; }
        if ($SLOW) {$nevt = 500; }

#my $CUR_DIR = cwd();

        my ($DIR1,$DIR2) = split /:/, $COMPARE_DIRS; 
	print "INFO: Comparing results in $DIR1 with those in $DIR2\n";

        # make comparision dir
        my $compStr1 = basename ($DIR1);
        my $compStr2 = basename ($DIR2);
        $COMP_DIR = "compare_$compStr1\_vs_$compStr2"; 
        mkdir($COMP_DIR) unless(-d $COMP_DIR);

        # go to comparision dir
	chdir $COMP_DIR;
	system "pwd";

	$ours = "../$DIR1/$WORK_DIR/test_0/L1Ntuple.root";
	$theirs = "../$DIR2/$WORK_DIR/test_0/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x '$ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/macros/NtupleDiff.C(\"reemul\",\"$ours\",\"$theirs\")'");

	$ours = "../$DIR1/$WORK_DIR/test_1/L1Ntuple.root";
	$theirs = "../$DIR2/$WORK_DIR/test_1/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x '$ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/macros/NtupleDiff.C(\"mc\",\"$ours\",\"$theirs\")'");

	$ours = "../$DIR1/$WORK_DIR/test_2/L1Ntuple.root";
	$theirs = "../$DIR2/$WORK_DIR/test_2/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x '$ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/macros/NtupleDiff.C(\"raw2016\",\"$ours\",\"$theirs\",\"l1UpgradeTree/L1UpgradeTree\",\"l1UpgradeTree/L1UpgradeTree\")'");

	$ours = "../$DIR1/$WORK_DIR/test_5/L1Ntuple.root";
	$theirs = "../$DIR2/$WORK_DIR/test_5/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x '$ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/macros/NtupleDiff.C(\"reemul2016\",\"$ours\",\"$theirs\")'");

	$ours = "../$DIR1/$WORK_DIR/test_6/L1Ntuple.root";
	$theirs = "../$DIR2/$WORK_DIR/test_6/L1Ntuple.root";
	if (! -e $ours)   { print "ERROR: could not find file $ours\n"; exit(1); }
	if (! -e $theirs) { print "ERROR: could not find file $theirs\n"; exit(1); }
	print "$ours\n";
	print "$theirs\n";;
	$status = long_command("root -b -q -x '$ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/macros/NtupleDiff.C(\"reemul2017\",\"$ours\",\"$theirs\")'");


	# this is a hack until L1T uGT output goes into L1TNtuple:
        system "sed -n \'/L1T menu Name/,/Final OR Count/p\' ../$DIR1/$WORK_DIR/test_0/CMSRUN.log > menu_a.txt";
        system "sed -n \'/L1T menu Name/,/Final OR Count/p\' ../$DIR2/$WORK_DIR/test_0/CMSRUN.log > menu_b.txt";
	print "INFO:  diff of menu summary of $compStr1 vs $compStr2 follows:\n";
	system "echo \'diff l1t menu with $nevt events of MC\' > diff_menu_a_vs_menu_b.txt\n";
	system "echo \'< $compStr1\' >> diff_menu_a_vs_menu_b.txt\n";
	system "echo \'> $compStr2\' >> diff_menu_a_vs_menu_b.txt\n";
	system "echo \'---------------------------------------------------------' >> diff_menu_a_vs_menu_b.txt\n";
	system "sleep 1";
	system "diff menu_a.txt menu_b.txt >> diff_menu_a_vs_menu_b.txt\n";
	$status = long_command("bash $ENV{CMSSW_BASE}/src/L1Trigger/L1TCommon/scripts/makeHtml.sh $compStr1 $compStr2");

        # go back to original dir
	chdir "..";
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

    if (! ($RECYCLE || $REDO)){
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
	    print "ERROR: --recycle or --redo specified but $WORK_DIR does not exist yet....\n";
	    return;
	}
    }


    $start_time = time();

    chdir $WORK_DIR;
    $PWD = `pwd`;
    chomp $PWD;
    $LOG = "$PWD/$MAIN_LOG";

    # fork off and die:
    my $pid = fork();
    die "$0: fork: $!" unless defined $pid;
    if ($pid) {
	print "INFO: Coffee Time!!!\n";
	if ($SLOW){ print "INFO:  Slow mode specified... better make it two!\n"; }
	print "INFO: You can view progress by running:\n";
	print "tail -f $LOG\n";
	exit(0);
    }    
    open STDOUT,">",$LOG or die $!;
    open STDERR,">",$LOG or die $!;

    print "INFO: launching $NUM_JOBS test jobs...\n" if (! $SINGLE );
    $PIDS   = ();
    $IJOB   = ();
    for ($ijob = 0; $ijob < $NUM_JOBS; $ijob++){
	next if ($SINGLE && ($ijob != $SINGLE_JOB));	
	if ($REDO) {
	    $SUCCESSFILE = "test_$ijob/SUCCESS";
	    if (-e $SUCCESSFILE) { print "INFO:  tests job $ijob was already successful... skipping\n"; next; }
	    else { if (-e "test_$ijob") { system "rm -fr test_$ijob"; } }
	}
	print "INFO: launching job number $ijob\n";	
	my $pid = fork();
	die "$0: fork: $!" unless defined $pid;

	if ($pid) {
	    print "INFO: child process $pid launched...\n";
	    push @PIDS, $pid;
	    push @IJOB, $ijob;
	} else {
	    run_job($ijob);
	}	
    }
 
    sub kill_children {
	foreach $pid (@PIDS) {
	    print "ERROR: killing child process $pid\n";
	    kill HUP => $pid;
	}	
	print "ERROR: exiting before all jobs complete..\n";
    };

    $SIG{ALRM} = sub {
	print "ERROR: timeout waiting for child processes...\n"; 
	kill_children();
	summary();
	exit 0;
    };   
    alarm $TIMEOUT;

    while($#PIDS >= 0){
	if (-e $DIE_FILE){
	    print "INFO: received DIE command... killing child processes and exiting...\n";
	    system "rm $DIE_FILE";
	    kill_children();
	    summary();
	    exit 0;
	} 

	for ($i=0; $i <= $#PIDS; $i++){
	    #print "DEBUG: $i\n";
	    $pid = waitpid ($PIDS[$i], &POSIX::WNOHANG);
	    #print "DEBUG: pid $pid\n";
	    if ($pid > 0){
		$SUCCESSFILE = "test_$IJOB[$i]/SUCCESS";
		if (-e $SUCCESSFILE) { print "INFO:  job $IJOB[$i] (pid $pid) has exited with status:  SUCCESS\n"; }
		else { print "INFO:  job $i (pid $pid) has exited with status:  FAILURE\n"; }
		splice(@PIDS, $i, 1);
		splice(@IJOB, $i, 1);
		$i--;
	    }
	} 
	system "sleep 1"
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
	next if ($SINGLE && ($ijob != $SINGLE_JOB));	
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
	if ($FAST || $DRYRUN || $RECYCLE || $REDO || $SINGLE){
	    print "STATUS:  results not sufficient for greenlight to commit due to --fast, --dryrun, --redo, --single, or --recycle option\n";
	} 
	else {
	    print "STATUS:  you have GREEN LIGHT to commit your L1T code!\n";
	    system "touch GREEN_LIGHT";
	}
    }
}
