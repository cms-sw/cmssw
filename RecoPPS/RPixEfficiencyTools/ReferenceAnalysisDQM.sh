#!/bin/bash
if [ $# -ne 3 ]
then
    echo "Usage: EfficiencyAnalysisDQM.sh <path to input file> <output directory path> <reference file path>"
else
	cmsRun python/ReferenceAnalysisDQMWorker_cfg.py sourceFileList=$1 outputFileName=tmp.root efficiencyFileName=$3
	cmsRun python/ReferenceAnalysisDQMHarvester_cfg.py inputFileName=tmp.root outputDirectoryPath=$2
	#rm tmp.root
fi