#!/bin/bash
function die { echo $1: status $2; exit $2; }

echo -e "Testing help functions"
printPixelLayersDisksMap --help  || die 'failed running printPixelLayersDisksMap --help' $?
printPixelROCsMap --help || die 'failed running printPixelROCsMap --help' $?
printPixelTrackerMap --help || die 'failed running printPixelTrackerMap --help' $?
printStripTrackerMap --help || die 'failed running printStripTrackerMap --help' $?
echo -e "\n"
testPixelFile=$CMSSW_BASE/src/SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt
[ -e $testPixelFile ] || testPixelFile=$CMSSW_RELEASE_BASE/src/SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt
# Store the first 50 elements of the first column in a variable
testPixelDetids=$(head -n 50 "$testPixelFile" | cut -d ' ' -f 1 | paste -sd ' ' -)

echo "Using the following pixel DetIds:" $testPixelDetids
echo -e "\n"
echo -e "==== Testing printPixelLayersDisksMap"
printPixelLayersDisksMap --input-file $testPixelFile || die 'failed printPixelLayersDisksMap --input-file' $?
printPixelLayersDisksMap $testPixelDetids || die 'failed printPixelLayersDisksMap $testPixelDetids' $?
echo -e "\n"
echo -e "==== Testing printPixelROCsMap"
printPixelROCsMap --input-file $testPixelFile || die 'failed printPixelROCsMap --input-file' $?
printPixelROCsMap $testPixelDetids || die 'failed printPixelROCsMap $testPixelDetids' $?
printPixelROCsMap $testPixelDetids --region barrel || die 'failed printPixelROCsMap $testPixelDetids --barrel' $?
printPixelROCsMap $testPixelDetids --region forward || die 'failed printPixelROCsMap $testPixelDetids --forward' $?
printPixelROCsMap $testPixelDetids --region full || die 'failed printPixelROCsMap $testPixelDetids --full' $?
echo -e "\n"
echo -e "==== Testing printPixelTrackerMap"
printPixelTrackerMap --input-file $testPixelFile || die 'failed printPixelTrackerMap --input-file' $?
printPixelTrackerMap $testPixelDetids || die 'failed printPixelTrackerMap $testPixelDetids' $?
echo -e "\n"
testStripFile=$CMSSW_BASE/src/CalibTracker/SiStripCommon/data/SiStripDetInfo.dat
[ -e $testStripFile ] || testStripFile=$CMSSW_RELEASE_BASE/src/CalibTracker/SiStripCommon/data/SiStripDetInfo.dat
# Store the first 50 elements of the first column in a variable
testStripDetids=$(head -n 50 "$testStripFile" | cut -d ' ' -f 1 | paste -sd ' ' -)

echo "Using the following strip DetIds:" $testStripDetids
echo -e "\n"
echo -e "==== Testing printStripTrackerMap"
printStripTrackerMap --input-file $testStripFile || die 'failed printStripTrackerMap --input-file' $?
printStripTrackerMap $testStripDetids || die 'failed printStripTrackerMap $testPixelDetids' $?
