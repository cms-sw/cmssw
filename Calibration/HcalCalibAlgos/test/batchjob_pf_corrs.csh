#!/bin/csh
#BSUB -q 1nh
##BSUB -J jfp

echo 'Start'

cmsrel CMSSW_3_1_4
cd CMSSW_3_1_4/src
cmsenv
cvs co Calibration/HcalCalibAlgos
scram b
cd Calibration/HcalCalibAlgos/test

set respcorrdir=/afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_4/src/Calibration/HcalCalibAlgos/data

# if you want to validate your own calibration, copy it to data/ from your local place: 
#cp $respcorrdir/calibConst_IsoTrk_testCone_26.3cm.txt ../data/response_corrections.txt
#cp $respcorrdir/HcalPFCorrs_v2.00_mc.txt ../data

cat > pfcorrs.py <<@EOF

import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalPFCorrsCulculation")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(

'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_${1}.root'

     )
)

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True)
)

process.load("Calibration.HcalCalibAlgos.pfCorrs_cfi")
process.hcalRecoAnalyzer.outputFile = cms.untracked.string("HcalCorrPF_${1}.root")
process.hcalRecoAnalyzer.ConeRadiusCm = cms.untracked.double(26.3)

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V5::All')
process.prefer("GlobalTag")

process.es_ascii2 = cms.ESSource("HcalTextCalibrations",
    appendToDataLabel = cms.string('recalibrate'),
    input = cms.VPSet(
     cms.PSet(
      object = cms.string('RespCorrs'),
      file = cms.FileInPath('Calibration/HcalCalibAlgos/data/response_corrections.txt')
             ),
     cms.PSet(
      object = cms.string('PFCorrs'),
      file = cms.FileInPath('Calibration/HcalCalibAlgos/data/HcalPFCorrs_v2.00_mc.txt')
#      file = cms.FileInPath('Calibration/HcalCalibAlgos/data/HcalPFCorrs_v1.03_mc.txt')
             )
       )
)


#process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
#process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")
process.p = cms.Path(process.hcalRecoAnalyzer)

#-----------
@EOF

cmsRun pfcorrs.py

set outdir=/afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_4/src/Calibration/HcalCalibAlgos/test

cp HcalCorrPF_*.root $outdir/

cd ../../../../../

rm -rf CMSSW*
