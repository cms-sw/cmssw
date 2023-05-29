#!/bin/bash
cd $CMSSW_BASE/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
export X509_USER_PROXY=.user_proxy
eval `scram runtime -sh`
cd workDirectory
./cmsRun validation_cfg.py config=validation.json
