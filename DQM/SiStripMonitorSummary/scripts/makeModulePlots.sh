#!/bin/bash

outputroot=/tmp/SiStripModules_$9.root
pedmon=False
noisemon=False
if [ $(( $3 & 1 )) -eq 1 ]; then
    noisemon=True
fi
if [ $(( $3 & 2 )) -eq 2 ]; then
    pedmon=True
fi
cmsRun $CMSSW_BASE/src/DQM/SiStripMonitorSummary/test/DBReader_conddbmonitoring_singlemodule_cfg.py logDestination=cout outputRootFile=$outputroot moduleList_load=$2 globalTag=$4 connectionString=$5 tagName=$6 recordName=$7 runNumber=$1 PedestalMon=$pedmon NoiseMon=$noisemon gainNorm=$8 

makeModulePlots $outputroot $2 $3 "/tmp/" $9