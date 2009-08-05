#!/bin/csh
#BSUB -q 1nh
##BSUB -J jfp

echo 'Start'
#export STAGE_SVCCLASS cmscaf

cmsrel CMSSW_3_1_0_pre10
cd CMSSW_3_1_0_pre10/src
cmsenv
cvs co Calibration/HcalCalibAlgos
scram b
cd Calibration/HcalCalibAlgos/test

set respcorrdir=/afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_0_pre10/src/Calibration/HcalCalibAlgos/data

# if you want to validate your own calibration, copy it to data/ from your local place: 
cp $respcorrdir/response_corrections.txt ../data/response_corrections.txt

cat > validator.py <<@EOF

import FWCore.ParameterSet.Config as cms

process = cms.Process("Validator")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_31X::All"

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")
process.load("Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi")
#process.IsoProd.SkipNeutralIsoCheck = cms.untracked.bool(True)
process.IsoProd.MinTrackP = cms.double(5.0)

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(5000)

process.load("Calibration.HcalCalibAlgos.calib_validator_cfi")
process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_XX.root")
process.ValidationIsoTrk.calibFactorsFileName = cms.string("Calibration/HcalCalibAlgos/data/response_corrections.txt")
process.ValidationIsoTrk.AxB = cms.string("3x3")
process.ValidationIsoTrk.takeAllRecHits = cms.untracked.bool(False)

#process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_10_${1}.root")
process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_50_${1}.root")
#process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_100_${1}.root")
#process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_300_${1}.root")

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(

'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_${1}.root'
#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_HEZS8_${1}.root'

#'rfio:/castor/cern.ch/user/a/abdullin/pi300_fullproduction_310pre10/pi300_${1}.root'

     )
)
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True)
)
process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")
process.p = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.ValidationIsoTrk)
#process.p = cms.Path(process.ValidationIsoTrk)

#-----------
@EOF

cmsRun validator.py

set outdir=/afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_0_pre10/src/Calibration/HcalCalibAlgos/test
#set outdir=/castor/cern.ch/user/a/andrey/pi50_310_pre10
#set outdir=/castor/cern.ch/user/a/andrey/pi300_310_pre10


cp ValidFile_*.root $outdir/
#rfcp ValidFile_*.root $outdir/

cd ../../../../../

rm -rf CMSSW*
