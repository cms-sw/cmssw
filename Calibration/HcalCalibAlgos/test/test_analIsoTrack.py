import FWCore.ParameterSet.Config as cms

process = cms.Process("ANTEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Calibration.HcalCalibAlgos.isoTrkCalib_cfi")

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
'file:/afs/cern.ch/user/s/sergeant/scratch0/2008/myRawToReco_IsoTr_FullFED.root'
#        'rfio:/castor/cern.ch/user/s/safronov/forIsoTracksFromReco.root'
)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)

process.pts = cms.EDFilter("PathTimerInserter")

process.PathTimerService = cms.Service("PathTimerService")

process.AnalIsoTrTest = cms.Path(process.isoTrkCalib)


