#!/bin/csh

set workDir=$ANALYZEDIRECTORY


echo "Beginning condor_rates.csh"

echo "-------------------------------"
echo "Current Directory: "
pwd
echo "-------------------------------"

source /uscmst1/prod/sw/cms/setup/cshrc prod
setenv SCRAM_ARCH slc5_amd64_gcc462

cd $workDir

eval `scram runtime -csh`
source setup.csh

echo "-------------------------------"
echo "Working Directory: "
pwd
echo "-------------------------------"

cd -
echo "Condor Directory: "
pwd


echo "Submitting job on `date`" 

@ pid = $argv[1] + 1
#set infiles = `ls /eos/uscms/store/user/ingabu/TMD/MinBias8TeVNtuples/`
set infiles = `ls /pnfs/cms/WAX/11/store/user/ingabu/TMD/QCD/QCD_80to120_TuneZ2star_8TeV-pythia6/`
setenv INFILE $infiles[$pid] 

echo $INFILE

$workDir/OHltRateEff $workDir/hltmenu_prescales.cfg

rm -f *.tex
mv hltmenu_8TeV_7.0e33_20130513_QCD.root hltmenu_8TeV_7.0e33_20130513_QCD_$pid.root
mv hltmenu_8TeV_7.0e33_20130513_QCD.twiki hltmenu_8TeV_7.0e33_20130513_QCD_$pid.twiki
mv Dataset_7e33V2_2012_correlations.root Dataset_7e33V2_2012_correlations_$pid.root

echo "Job finished on `date`" 
