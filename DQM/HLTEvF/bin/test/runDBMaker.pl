#!/usr/bin/env perl

use File::Copy;
# This scrits must be used ./runNtpMakerCAF.pl RunNumber
# It looks for streamer files from run with name $GLOBAL following
# the CMS conventions for the path name.
#Jobs are submitted on the CAF.

 
#------Configure here ---------------------------------------
$queue = "cmscaf1nh";
$curDir=`pwd`;
chomp $curDir;
$pathToFiles="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/PromptReco";
#$HLTPATH = "HLT_L1MuOpen"; $thr =30.;
#$HLTPATH = "HLT_L1Mu20"; $thr =30.;
#$HLTPATH = "HLT_L2Mu3_NoVertex"; $thr =30.;
$HLTPATH = "HLT_L2Mu9"; $thr =30.;
#$HLTPATH = "HLT_L2Mu11"; $thr =30.;
#$HLTPATH = "HLT_Photon10_L1R"; $thr =25.;
#$HLTPATH = "HLT_Photon15_L1R"; $thr =35.;
#$HLTPATH = "HLT_Photon15_L1R_TrackIso_L1R"; $thr =35.;
#$HLTPATH = "HLT_Photon15_LooseEcalIso_L1R"; $thr =35.;


#client
#$histoPath = "FourVector/client/$HLTPATH/custom-eff";
#$name = "$HLTPATH\_wrt__l1Et_Eff_OnToL1_UM";
#$name = "$HLTPATH\_wrt__offEt_Eff_L1ToOff_UM";
#$name = "$HLTPATH\_wrt__offEt_Eff_OnToOff_UM";

#source
$histoPath = "FourVector/source/$HLTPATH";
$name = "$HLTPATH\_wrt__NL1";
#$name = "$HLTPATH\_wrt__NOn";
#$name = "$HLTPATH\_wrt__NOff";
#$name = "$HLTPATH\_wrt__l1EtL1";
#$name = "$HLTPATH\_wrt__offEtOff";
#$name = "$HLTPATH\_wrt__onEtOn";


$detid =2;
$par1 ="rms";
$par2 ="usrMean";
$par3 ="plateau";
#-------------------------------------------------------------


system("rm -f $name\_dbfile.db\n");
print("rm -f $name\_dbfile.db\n");
mkdir($name);
copy("submitMacro.ch", $name);
chdir($name);
system("chmod +x submitMacro.ch\n");
#read file list
$line=0;
while(<>)
{
$run = $_;
chomp($run);
$A = substr($run,0,3);
#$ZERO = "0";
#$A = $ZERO.$A;
$B = substr($run,3);
#$file = "$pathToFiles/$A/$B/DQM_V0001_R000$run\__Cosmics__Commissioning09-PromptReco-v7__RECO.root";
@v = glob("$pathToFiles/$A/$B/DQM_V0001_R000$run\__Cosmics__*");
$file = @v[0];
print "$file\n";

$line++;




open CFGFILE, "> historyClient_$run\_$name\_cfg.py";

print CFGFILE "import FWCore.ParameterSet.Config as cms\n";
print CFGFILE "\n";
print CFGFILE "process = cms.Process(\"PWRITE\")\n";
print CFGFILE "process.MessageLogger = cms.Service(\"MessageLogger\",\n";
print CFGFILE "			destinations = cms.untracked.vstring('readFromFile_$run'),\n";
print CFGFILE "			readFromFile_$run = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),\n";
print CFGFILE "			debugModules = cms.untracked.vstring('*')\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.maxEvents = cms.untracked.PSet(\n";
print CFGFILE "			input = cms.untracked.int32(1))\n";
print CFGFILE "\n";
print CFGFILE "process.source = cms.Source(\"EmptySource\",\n";
print CFGFILE "			timetype = cms.string(\"runnumber\"),\n";
print CFGFILE "			firstRun = cms.untracked.uint32(1),\n";
print CFGFILE "			lastRun  = cms.untracked.uint32(1),\n";
print CFGFILE "			interval = cms.uint32(1)\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.load(\"DQMServices.Core.DQM_cfg\")\n";
print CFGFILE "\n";
print CFGFILE "process.PoolDBOutputService = cms.Service(\"PoolDBOutputService\",\n";
print CFGFILE "			BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),\n";
print CFGFILE "			outOfOrder = cms.untracked.bool(True),\n";
print CFGFILE "			DBParameters = cms.PSet(\n";
print CFGFILE "messageLevel = cms.untracked.int32(2),\n";
print CFGFILE "authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')\n";
print CFGFILE "),\n";
print CFGFILE "\n";
print CFGFILE "			timetype = cms.untracked.string('runnumber'),\n";
print CFGFILE "			connect = cms.string('sqlite_file:$name\_dbfile.db'),\n";
print CFGFILE "			toPut = cms.VPSet(cms.PSet(\n";
print CFGFILE "			record = cms.string(\"HDQMSummary\"),\n";
print CFGFILE "			tag = cms.string(\"HDQM_test\")\n";
print CFGFILE ")),\n";
print CFGFILE "			logconnect = cms.untracked.string(\"sqlite_file:$name\_log.db\") \n";
print CFGFILE "			)\n";
print CFGFILE "\n";
print CFGFILE "process.hltDQMHistoryPopCon = cms.EDAnalyzer(\"HLTDQMHistoryPopCon\",\n";
print CFGFILE "			record = cms.string(\"HDQMSummary\"),\n";
print CFGFILE "			loggingOn = cms.untracked.bool(True),\n";
print CFGFILE "			SinceAppendMode = cms.bool(True),\n";
print CFGFILE "			Source = cms.PSet(since = cms.untracked.uint32($run),\n";
print CFGFILE "			debug = cms.untracked.bool(False))\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.HLTHistoryDQMService = cms.Service(\"HLTHistoryDQMService\",\n";
print CFGFILE "			RunNb = cms.uint32($run),\n";
print CFGFILE "			accessDQMFile = cms.bool(True),\n";
print CFGFILE "			FILE_NAME = cms.untracked.string(\"$file\"),\n";
print CFGFILE "			ME_DIR = cms.untracked.string(\"Run $run/HLT/Run summary/$histoPath/\"),\n";
print CFGFILE "			threshold = cms.untracked.double($thr),\n";
print CFGFILE "			histoList = cms.VPSet(\n";
print CFGFILE "\n";
print CFGFILE "cms.PSet( keyName = cms.untracked.string('$name'), quantitiesToExtract = cms.untracked.vstring(\"stat\")),\n";
print CFGFILE "cms.PSet( keyName = cms.untracked.string('$name'), quantitiesToExtract = cms.untracked.vstring(\"plateau\"))\n";
print CFGFILE "\n";
print CFGFILE ")\n";
print CFGFILE ")\n";
print CFGFILE "process.p = cms.Path(process.hltDQMHistoryPopCon)\n";
print CFGFILE "\n";
close CFGFILE ;

print "cmsRun historyClient_$run\_$name\_cfg.py\n";
system("cmsRun historyClient_$run\_$name\_cfg.py >& $name.log\n");
#system("bsub -J $run\_$name -q $queue cmsRun historyClient_$run\_$name\_cfg.py\n");

}
system("cmscond_list_iov -c sqlite_file:$name\_dbfile.db -t HDQM_test\n");


print "End submission...\n";

