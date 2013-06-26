#!/bin/bash

# This script takes a file from the argument and runs the BadAPVIndentifier
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
    echo "Example: ./startBadAPVIdentifier.sh somefile.root"
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
echo $DBFileName
cp $template_dir/dbfile_empty.db ./$DBFileName

echo "Creating BadAPVIdentifier config from template"
cat $template_dir/template_SiStripQualityBadAPVIdentifierRoot_cfg.py |sed -e "s@insertRun@$run@g" -e "s@dbfile.db@$DBFileName@" -e "s@insertInputDQMfile@$file@" > log/SiStripQualityBadAPVIdentifierRoot_${run}_cfg.py

echo "Starting cmsRun BadAPVIdentifier"
cmsRun log/SiStripQualityBadAPVIdentifierRoot_${run}_cfg.py > log/SiStripQualityBadAPVIdentifierRoot_${run}.log

echo "Creating SiStripQualityStatistics_offline config from template"
cat $template_dir/template_SiStripQualityStatistics_offline_cfg.py |sed -e "s@insertRun@$run@g" -e "s@dbfile.db@$DBFileName@" -e "s@inputTag@HotStrips@" > log/SiStripQualityStatistics_offline_${run}_BadAPVs_cfg.py

echo "Starting cmsRun SiStripQualityStatistics_offline"
cmsRun log/SiStripQualityStatistics_offline_${run}_BadAPVs_cfg.py > out_${run}_BadAPVs.tmp

cat out_${run}_BadAPVs.tmp  | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;print "";} if(doprint==1) print $0}' > results/BadAPVs_${run}.txt

# Cleaning up and moving results to propper dir
mv out_${run}_* results/
mv $DBFileName results/
mv BadAPVOccupancy_${run}.root results/

echo "Run $run finished"
