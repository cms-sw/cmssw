#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING GCP Ntuple creation"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/Ntuples/SURun3_1
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for SURun3 alignment" $?
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/Ntuples/ideal_1
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for ideal alignment" $?

echo "TESTING GCP comparison tree creation"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPtree/
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP comparison tree creation step" $?

echo "TESTING GCP cpp plots"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPcpp/
./GCP validation.json -v || die "Failure running GCP cpp plots" $?

echo "TESTING GCP python plots"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPpython/
./GCPpyPlots.py validation.json || die "Failure running GCP python plots" $?
