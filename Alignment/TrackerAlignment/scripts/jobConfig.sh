#!/bin/bash

cd $CMSSW_BASE/src/Alignment/TrackerAlignment/test 
eval `scramv1 runtime -sh`
cd -
cmsRun $CMSSW_BASE/src/Alignment/TrackerAlignment/test/cosmicRateAnalyzer_cfg.py

rfcp Cosmic_rate_tuple.root $CMSSW_BASE/src/Alignment/TrackerAlignment/test

