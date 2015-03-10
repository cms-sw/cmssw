#!/bin/bash

# Usage: Pass in your EcalTimingAnalysis-created root file (containing the TTree)
#        and a name for the directory where the new calibrations will be created
#
# Example: ./makeCalibs.sh myTimingAnalysisRootFile.root newSplash2009TimingSettings

# set up cmssw area
eval `scramv1 run -sh`

# make and cd into the new directory
mkdir $2
cd $2

# run binaries to make the raw crystal calibrations
echo "Creating raw crystal calibrations for EB..."
createTimingCalibsEB $1
echo "Now creating raw crystal calibrations for EE..."
createTimingCalibsEE $1

# Make and run cfgs for creating HW settings and subtracting
# those from the raw crystal calibrations calculated above
# Must do subtractTowerAvgForOfflineCalibs = cms.untracked.bool(False)
# in the createAvgs module if we don't want to subtract the HW values
# Can do this for EB/EE at the same time now

# Combine EB/EE calibs
cat timingCalibsEB.calibs.txt timingCalibsEE.calibs.txt > timingCalibsEcal.calibs.txt

echo "Creating cfg and cmsRunning to make HW/SW settings..."

cat > ecalcreateavgTTtimes_cfg.py <<EOF

import FWCore.ParameterSet.Config as cms

process = cms.Process("createTTAvgsEcal")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.createAvgs = cms.EDAnalyzer('EcalCreateTTAvgTimes',
    timingCalibFile = cms.untracked.string('timingCalibsEcal.calibs.txt')
)

process.p = cms.Path(process.createAvgs)

EOF

cmsRun ecalcreateavgTTtimes_cfg.py

