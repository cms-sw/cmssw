#!/bin/csh
#BSUB -q 1nh
##BSUB -J jfp

echo 'Start'
#export STAGE_SVCCLASS cmscaf

cmsrel CMSSW_3_1_2
cd CMSSW_3_1_2/src
cmsenv
cvs co Calibration/HcalCalibAlgos
cp /afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_2/src/Calibration/HcalCalibAlgos/src/HcalIsoTrkAnalyzer.cc  Calibration/HcalCalibAlgos/src
scram b
cd Calibration/HcalCalibAlgos/test


cat > myjob.py <<@EOF

import FWCore.ParameterSet.Config as cms

process = cms.Process("ANTEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_31X::All'
process.GlobalTag.globaltag = 'MC_31X_V5::All'

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.load("Calibration.HcalCalibAlgos.isoAnalyzer_cfi")
process.isoAnalyzer.AxB = cms.string("7x7")
#outputFileName = cms.string("test_IsoAn.root"),

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames =
    cms.untracked.vstring(

    'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_${1}.root',
    #'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_${1}.root',
    #'file:/afs/cern.ch/user/s/sergeant/scratch0/2008/myRawToReco_IsoTr_FullFED.root'
    
    #'file:/afs/cern.ch/user/s/sergeant/scratch0/2009/CMSSW_3_1_0/src/Configuration/GenProduction/test/ALCARECOHcalCalIsoTrk.root'

    )
    )

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")

#Use this Path to run the code on RECO data sets (such as single pions produced by Salavat):
process.AnalIsoTrTest = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.isoAnalyzer)

#Use this Path instead to run it on ALCARECO format data:
#process.AnalIsoTrTest = cms.Path(process.isoAnalyzer)


#-----------
@EOF

cmsRun myjob.py

set outdir=/afs/cern.ch/user/a/andrey/public/isoTrkCalibFiles
#set outdir=/afs/cern.ch/user/a/andrey/scratch1/CMSSW_3_1_0_pre10/src/Calibration/HcalCalibAlgos/test
#set outdir=/castor/cern.ch/user/a/andrey/

cp rootFile.root $outdir/rootFile_${1}.root
#rfcp rootFile.root $outdir/rootFile_${1}.root


cd ../../../../../

rm -rf CMSSW*
