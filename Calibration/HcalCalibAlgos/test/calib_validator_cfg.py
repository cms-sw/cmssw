import FWCore.ParameterSet.Config as cms

process = cms.Process("Validator")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_31X::All'
#process.GlobalTag.globaltag = 'IDEAL_V9::All'

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")
#process.IsoProd.SkipNeutralIsoCheck = cms.untracked.bool(True)
process.IsoProd.MinTrackP = cms.double(5.0)

process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(200)

#process.load("UserCode.AndreyPozdnyakov.validator_cfi")
process.load("Calibration.HcalCalibAlgos.calib_validator_cfi")
process.ValidationIsoTrk.outputFileName = cms.string("ValidFile_XX.root")
#process.ValidationIsoTrk.calibFactorsFileName = cms.string("Calibration/HcalCalibAlgos/data/calibFactors_merged_3x3.txt")
process.ValidationIsoTrk.calibFactorsFileName = cms.string("Calibration/HcalCalibAlgos/data/calibConst_IsoTrk_test071809.txt")
#process.ValidationIsoTrk.calibFactorsFileName = cms.string("UserCode/AndreyPozdnyakov/data/calibConst_IsoTrk_test071309.txt")
process.ValidationIsoTrk.AxB = cms.string("3x3")
process.ValidationIsoTrk.takeAllRecHits = cms.untracked.bool(True)

process.load("HLTrigger.Timer.timer_cfi")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#'rfio:/castor/cern.ch/user/a/abdullin/single_pi_full/pi10_1.root'
#'rfio:/castor/cern.ch/user/a/abdullin/single_pi_full/pi50_2.root',
#'rfio:/castor/cern.ch/user/a/abdullin/single_pi_full/pi50_3.root',
#'rfio:/castor/cern.ch/user/a/abdullin/single_pi_full/pi100_1.root'

'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_1.root',
#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_310pre10/pi50_2.root'
#'rfio:/castor/cern.ch/user/a/abdullin/pi300_fullproduction_310pre10/pi300_1.root'


#'/store/mc/CSA08/HCalIsoTracks50/ALCARECO/1PB_V2_RECO_HcalCalIsoTrkNoHLT_v1/0033/001CFAC4-1125-DD11-A5F5-000423D944F8.root',

    )
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
process.pts = cms.EDFilter("PathTimerInserter")
process.PathTimerService = cms.Service("PathTimerService")


#process.p = cms.Path(process.IsoProd*process.ValidationIsoTrk)
process.p = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT*process.ValidationIsoTrk)
#process.p = cms.Path(process.ValidationIsoTrk)



