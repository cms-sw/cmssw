#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING GCP Ntuple creation"
pushd test_yaml/GCP/GCPdetUnits/Ntuples/SURun3_1
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for SURun3 alignment" $?
popd

pushd test_yaml/GCP/GCPdetUnits/Ntuples/ideal_1
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP Ntuple for ideal alignment" $?
popd

echo "TESTING GCP comparison tree creation"
pushd test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPtree/
./cmsRun validation_cfg.py config=validation.json || die "Failure running GCP comparison tree creation step" $?
popd

echo "TESTING GCP cpp plots"
pushd test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPcpp/
./GCP validation.json -v || die "Failure running GCP cpp plots" $?
popd

echo "TESTING GCP python plots"
pushd test_yaml/GCP/GCPdetUnits/SURun3vsIdeal/1_vs_1/GCPpython/
./GCPpyPlots.py validation.json || die "Failure running GCP python plots" $?
popd
