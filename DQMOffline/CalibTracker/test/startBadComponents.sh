#!/bin/bash

# This script takes a root file from the command line argument and runs 
# first the BadAPVIndentifier and then the HotStripIdentification.
# Results will be found in a result subdir from where the script is run
# All results will be identified by "_runnumber" added to the filename.
# The runnumber is derived from the inputfilename.

RunDIR=$(pwd)

# Check if the CMSSW environment has been set, otherwise try do do it
if [ $(echo $SCRAMRT_SET | grep -c CMSSW) -eq 0 ]; then
    if [ $(echo $RunDIR | grep -c CMSSW) -gt 0 ]; then
	CMSSWDIR=$(echo $RunDIR | awk '{end=match($0,"src"); print substr($0,0,end+2)}')
	cd $CMSSWDIR
	eval `scramv1 runtime -sh`
	cd $RunDIR
    else
	echo "CMSSW could not be initialized, exiting"
	exit 1
    fi
fi

CMSSWDIR=$CMSSW_BASE
template_dir=$CMSSWDIR/src/DQMOffline/CalibTracker/test

# Getting run number from filename (ARG #1)
if [ ${#1} -eq 0 ]; then
    echo "Please provide a filename as ARG1 for this script!"
    echo "Example: ./startBadComponents.sh somefile.root"
    exit 1
fi

file=$1
run=`echo $file| awk -F"R00" '{print $2}' | awk -F"_" '{print int($1)}'`
echo Filename:$file, Run: $run

# Create dir for log and config if not existing yet
if [ ! -d log ]; then 
    echo "Creating log dir"
    mkdir log; 
fi

# Create dir for results if not existing yet
if [ ! -d results ]; then 
    echo "Creating results dir"
    mkdir results; 
fi

DBFileName=dbfile_${run}.db
cp $template_dir/dbfile_empty.db ./$DBFileName

echo "Creating BadAPVIdentifier config from template"
cat $template_dir/template_SiStripQualityBadAPVIdentifierRoot_cfg.py |sed -e "s@insertRun@$run@g" -e "s@dbfile.db@$DBFileName@" -e "s@insertInputDQMfile@$file@" > log/SiStripQualityBadAPVIdentifierRoot_${run}_cfg.py

echo "Starting cmsRun BadAPVIdentifier"
cmsRun log/SiStripQualityBadAPVIdentifierRoot_${run}_cfg.py > log/SiStripQualityBadAPVIdentifierRoot_${run}.log

echo "Creating HotStripIdentification config from template"
cat $template_dir/template_SiStripQualityHotStripIdentifierRoot_cfg.py |sed -e "s@insertRun@$run@"  -e "s@insertInputDQMfile@$file@"  -e "s@dbfile.db@$DBFileName@" > log/SiStripQualityHotStripIdentifierRoot_${run}_cfg.py

echo "Starting cmsRun HotStripIdentification"
cmsRun log/SiStripQualityHotStripIdentifierRoot_${run}_cfg.py > log/SiStripQualityHotStripIdentifierRoot_$run.log

echo "Creating SiStripQualityStatistics_offline config from template"
cat $template_dir/template_SiStripQualityStatistics_offline_cfg.py |sed -e "s@insertRun@$run@" -e "s@dbfile.db@$DBFileName@" -e "s@inputTag@SiStripBadChannel_v1@" > log/SiStripQualityStatistics_offline_${run}_BadAPVs_cfg.py
cat $template_dir/template_SiStripQualityStatistics_offline_cfg.py |sed -e "s@insertRun@$run@" -e "s@dbfile.db@$DBFileName@" -e "s@inputTag@HotStrips@" > log/SiStripQualityStatistics_offline_${run}_HotStrips_cfg.py

echo "Starting cmsRun SiStripQualityStatistics_offline"
cmsRun log/SiStripQualityStatistics_offline_${run}_BadAPVs_cfg.py > out_${run}_BadAPVs.tmp
cmsRun log/SiStripQualityStatistics_offline_${run}_HotStrips_cfg.py > out_${run}_HotStrips.tmp

cat out_${run}_BadAPVs.tmp | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;print "";} if(doprint==1) print $0}' > results/BadAPVs_${run}.txt
cat out_${run}_HotStrips.tmp | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;print "";} if(doprint==1) print $0}' > results/HotStrips_${run}.txt

# Cleaning up and moving results to propper dir
mv out_${run}_* results/
mv $DBFileName results/
if [ -f BadAPVOccupancy_${run}.root ]; then
    mv BadAPVOccupancy_${run}.root results/
fi
if [ -f HotStripsOccupancy_${run}.root ]; then
    mv HotStripsOccupancy_${run}.root results/
fi

echo "Run $run finished"
