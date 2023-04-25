#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/DMR single configuration with json..."
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/DMR/single/TestSingleMC/unitTest/1/
./cmsRun validation_cfg.py config=validation.json || die "Failure running DMR single configuration with json" $?

echo "TESTING Alignment/DMR single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running DMR single configuration standalone" $?

echo "TESTING DMR merge step"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/DMR/merge/TestMergeMC/1/
./DMRmerge validation.json || die "Failure running DMR merge step" $?

echo "TESTING DMR trends"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/DMR/trends/TestTrendMC/
./DMRtrends validation.json --verbose || die "Failure running DMR trends" $?
 
echo "TESTING DMR averaged"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/DMR/averaged/TestAveragedMC/MC/
./mkLumiAveragedPlots.py validation.json || die "Failure running DMR averaged in merge mode" $?
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/DMR/averaged/TestAveragedMC/plots/
./mkLumiAveragedPlots.py validation.json || die "Failure running DMR averaged in plotting mode" $?
