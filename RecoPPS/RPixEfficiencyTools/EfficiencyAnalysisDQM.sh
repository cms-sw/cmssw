#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Usage: EfficiencyAnalysisDQM.sh <path to input file> <output directory path>"
else
	cmsRun python/EfficiencyAnalysisDQMWorker_cfg.py sourceFileList=$1 outputFileName=tmp.root bunchSelection=NoSelection
	cmsRun python/EfficiencyAnalysisDQMHarvester_cfg.py inputFileName=tmp.root outputDirectoryPath=$2
	rm tmp.root
fi