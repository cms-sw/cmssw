#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING GCP Ntuple creation"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/Ntuples/legacy_274094
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for legacy alignment" $?
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/Ntuples/prompt_274094
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for prompt alignment" $?

echo "TESTING GCP comparison tree creation"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/legacyVSprompt/274094_vs_274094/GCPtree/
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP comparison tree creation step" $?

echo "TESTING GCP cpp plots"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/legacyVSprompt/274094_vs_274094/GCPcpp/
./GCP validation.json || die "Failure running GCP cpp plots" $?

echo "TESTING GCP python plots"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/legacyVSprompt/274094_vs_274094/GCPpython/
./GCPpyPlots.py validation.json || die "Failure running GCP python plots" $?
