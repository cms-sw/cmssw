#!/usr/bin/env perl

die "*** Usage: ./rocProcessFile.perl <RunNo> <fileType> <filePath> *** " if ($#ARGV<2);

$runNo = $ARGV[0];
$fileType = $ARGV[1];
$filePath = $ARGV[2];

$baseDir = $ENV{'CMSSW_BASE'};
$workDir = $ENV{'PWD'};
$archiveDir = "/tmp/wfisher/";

print("*******************************************\n");
print("    Hcal DQM ROC File Processing Script\n");
print("*******************************************\n\n");
print("CMSSW_BASE=$baseDir\n");
print("Working Dir=$workDir\n");
print("Archive Dir=$archiveDir\n");

die "*** Filetype must be an integer between 0-2***\n  *FileType 0 = HcalTBData\n  *FileType 1 =PoolSource\n  *FileType 2 = StreamerFile\n" if($fileType!=0 && $fileType!=1 && $fileType!=2);

die "*** Your CMSSW_BASE environment variable is not set! *** " if($baseDir eq "");
die "*** Cannot write to $archiveDir! *** " if (!(-w $archiveDir));

if(index($filePath, "store")==-1){
    die "*** The file $filePath does not exist!! *** \n" if (!(-s $filePath));
    die "*** The file $filePath is not readable!! *** \n" if (!(-r $filePath));
}

$typeName = "HcalTBDataFile";
$fileName = "";
if($fileType==0){
    print("\n==>Processing $filePath as $typeName\n");
    $fileName = "'file:".$filePath."'";
}
if($fileType==1){
    $typeName = "PoolFile";
    print("\n==>Processing $filePath as $typeName\n");
    if(index($filePath, "store")!=-1){
	$fileName = "'".$filePath."'";
    }
    else{
	$fileName = "'file:".$filePath."'";
    }
}
if($fileType==2){
    $typeName = "StreamerFile";
    print("\n==>Processing $filePath as $typeName\n");
    if(index($filePath, "store")!=-1){
	$fileName = "'".$filePath."'";
    }
    else{
	$fileName = "'file:".$filePath."'";
    }
}

$configFile = $workDir."/rocProcessConfig.cfg";
$logFile = "run".$runNo.".log";

if(-e $logFile){
    system("rm -rf $logFile");
}

print("==>STDIO and STDERR will be logged in $logFile\n");

$transferLoc = sprintf("%09d", $runNo);
$transferLoc = $archiveDir.$transferLoc;

$rootOutput = sprintf("DQM_Hcal_R%09d.root",$runNo);
$htmlOutput = sprintf("DQM_Hcal_R%09d",$runNo);

print("==>All output will be archived in $transferLoc\n");

if (-d $transferLoc){
    die "*** Cannot write to $transferLoc! *** " if (!(-w $transferLoc));
}
if (!(-d $transferLoc)){
    system("/bin/mkdir $transferLoc");
}

open(FILE, ">$configFile");

print FILE <<EOS;

process HCALDQM = { 
    include "DQM/HcalMonitorModule/data/HcalMonitorModule.cfi"
	include "DQM/HcalMonitorClient/data/HcalMonitorClient.cfi"
	include "DQM/HcalMonitorModule/data/Hcal_FrontierConditions_GREN.cff"
	include "FWCore/MessageLogger/data/MessageLogger.cfi"
	
	include "EventFilter/HcalRawToDigi/data/HcalRawToDigi.cfi"
        include "RecoLocalCalo/HcalRecProducers/data/HcalSimpleReconstructor-hbhe.cfi"
        include "RecoLocalCalo/HcalRecProducers/data/HcalSimpleReconstructor-ho.cfi"
        include "RecoLocalCalo/HcalRecProducers/data/HcalSimpleReconstructor-hf.cfi"
	
	path p = { hcalDigis, horeco, hfreco, hbhereco, hcalMonitor, hcalClient, dqmEnv, dqmSaver }
    
    untracked PSet options = {
	untracked vstring Rethrow = { "ProductNotFound", "TooManyProducts", "TooFewProducts" }
    }
    untracked PSet maxEvents = {untracked int32 input = 201 }
    
    
    service = DaqMonitorROOTBackEnd{}
    
    #########################################################
    #### BEGIN DQM Online Environment #######################
    #########################################################
    # use include file for dqmEnv dqmSaver
    include "DQMServices/Components/test/dqm_onlineEnv.cfi"
	replace dqmSaver.fileName      = "Hcal"
	replace dqmEnv.subSystemFolder = "Hcal"
	
	# optionally change fileSaving  conditions
	// replace dqmSaver.prescaleLS =   -1
	// replace dqmSaver.prescaleTime = -1 # in minutes
	// replace dqmSaver.prescaleEvt =  -1
	// replace dqmSaver.saveAtRunEnd = true
	// replace dqmSaver.saveAtJobEnd = false
	# will add switch to select histograms to be saved soon
	
	#########################################################
	#### END ################################################
	#########################################################
	
EOS
close(FILE);
    
if($fileType==0){
    open(FILE, ">>$configFile");

print FILE <<EOS;

    source = HcalTBSource {
	untracked vstring fileNames = {
	    $fileName
	}
	untracked vstring streams = {
	    'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703','HCAL_DCC704',
	    'HCAL_DCC705','HCAL_DCC706','HCAL_DCC707','HCAL_DCC708','HCAL_DCC709',
	    'HCAL_DCC710','HCAL_DCC711','HCAL_DCC712','HCAL_DCC713','HCAL_DCC714',
	    'HCAL_DCC715','HCAL_DCC716','HCAL_DCC717','HCAL_DCC718','HCAL_DCC719',
	    'HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723','HCAL_DCC724',
	    'HCAL_DCC725','HCAL_DCC726','HCAL_DCC727','HCAL_DCC728','HCAL_DCC729',
	    'HCAL_DCC730','HCAL_DCC731'
	    }
    }
}
EOS
close(FILE);
}
if($fileType==1){
open(FILE, ">>$configFile");
print FILE <<EOS;

    source = PoolSource{
	untracked vstring fileNames = {
	    $fileName
	}
    }
}
EOS
close(FILE);
}
if($fileType==2){
open(FILE, ">>$configFile");
print FILE <<EOS;

    source = NewEventStreamFileReader {
	untracked vstring fileNames = {
	    $fileName
	}
    }
}
EOS
close(FILE);
}

system("cmsRun $configFile >& $logFile");

system("rm -rf DQM_Hcal_R000000000.root");
system("mv $configFile $transferLoc");
system("mv $logFile $transferLoc");
system("mv $rootOutput $transferLoc");
system("mv $htmlOutput $transferLoc");

