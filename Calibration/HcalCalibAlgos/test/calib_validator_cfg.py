import FWCore.ParameterSet.Config as cms

process = cms.Process("Validator")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'FIRSTCOLL::All'
#process.GlobalTag.globaltag = 'GR09_P_V6::All'
#process.GlobalTag.globaltag = 'MC_31X_V5::All'

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")
#process.IsoProd.SkipNeutralIsoCheck = cms.untracked.bool(True)
process.IsoProd.MinTrackP = cms.double(1.0)

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Calibration.HcalCalibAlgos.calib_validator_cfi")
process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_XX.root")
process.ValidationIsoTrk.calibFactorsFileName = cms.string("Calibration/HcalCalibAlgos/data/response_corrections.txt")
#process.ValidationIsoTrk.calibFactorsFileName =cms.string("Calibration/HcalCalibAlgos/data/calibConst_IsoTrk_testCone_26.3cm.txt")
process.ValidationIsoTrk.AxB = cms.string("Cone")
#process.ValidationIsoTrk.AxB = cms.string("3x3")
process.ValidationIsoTrk.calibrationConeSize = cms.double(26)
process.ValidationIsoTrk.takeAllRecHits = cms.untracked.bool(False)

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(20)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/FE5EDBBC-7DD9-DE11-9589-001A92971B64.root',     
#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/FAE7D85F-8AD9-DE11-A342-0026189438C4.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_FIRSTCOLL_v1/0083/F8EE1B4D-28D9-DE11-BF9C-00261894393B.root',

#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_3.root',

#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_1.root',
#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_2.root'
#'rfio:/castor/cern.ch/user/a/abdullin/pi300_fullproduction_310pre10/pi300_1.root'


    )
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")


process.p = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.ValidationIsoTrk)
#process.p = cms.Path(process.IsoProd*process.ValidationIsoTrk)
#process.p = cms.Path(process.ValidationIsoTrk)



