#!/bin/bash -ex

SCRAM_TEST_NAME=TestFWCoreSkeletons

# Create a new CMSSW dev area
OLD_CMSSW_BASE=$CMSSW_BASE
rm -rf $SCRAM_TEST_NAME
mkdir $SCRAM_TEST_NAME
cd $SCRAM_TEST_NAME
scram -a $SCRAM_ARCH project $CMSSW_VERSION
cd $CMSSW_VERSION/src
eval `scram run -sh`
git init

# Copy FWCore/Skeletons in cause unit test is run during PR tests which contains changes for FWCore/Skeletons
if [ -d $OLD_CMSSW_BASE/src/FWCore/Skeletons ] ; then
    mkdir FWCore
    cp -r $OLD_CMSSW_BASE/src/FWCore/Skeletons FWCore/Skeletons
    scram build
fi

# Create TestSubsystem for testing the scripts
mkdir TestSubsystem

# Test output of scripts that can be easily compiled
pushd TestSubsystem

mkdqmedanalyzer TestDQMAnalyzerStream example_stream -author "Test Author"
mkdqmedanalyzer TestDQMAnalyzerGlobal example_global -author "Test Author"
mkdqmedanalyzer -debug TestDQMAnalyzerStrDbg example_stream -author "Test Author"
mkdqmedanalyzer -debug TestDQMAnalyzerGlbDbg example_global -author "Test Author"

mkedanlzr TestEDAnalyzer -author "Test Author"
mkedanlzr TestEDAnalyzerHisto example_histo -author "Test Author"

mkedfltr TestEDFilter -author "Test Author"

mkedprod TestEDProducer -author "Test Author"
mkedprod TestEDProducerParticle example_myparticle -author "Test Author"

mkrecord TestRecord -author "Test Author"

popd
git add TestSubsystem
git commit -a -m "Add skeleton modules that can be easily compiled"
scram build

# Test output of scripts for which compilation would be difficult, but
# code-checks/code-format still works
pushd TestSubsystem

mkdatapkg TestDataPackage -author "Test Author"

mkskel TestSkeleton -author "Test Author"

popd
git add TestSubsystem
git commit -a -m "Other skeletong code"
scram b code-checks-all
scram b code-format-all
git diff
if [ $(git diff --name-only | grep TestSubsystem | wc -l) -gt 0 ] ; then
    echo "code-checks or code-format caused differences, skeleton templates need to be updated!"
    exit 1
fi

# Test output of scripts for which compilation, code-checks, and
# code-format would be difficult
pushd TestSubsystem
mkedlpr TestEDLooper -author "Test Author"
mkedlpr TestEDLooperRecData TestRecord TestData1 TestData2 -author "Test Author"

mkesprod TestESProducer -author "Test Author"
mkesprod TestESProducerRec TestRecord -author "Test Author"
mkesprod TestESProducerRecData TestRecord TestData -author "Test Author"
popd
