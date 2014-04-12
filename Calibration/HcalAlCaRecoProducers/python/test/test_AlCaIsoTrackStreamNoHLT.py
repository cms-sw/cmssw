import FWCore.ParameterSet.Config as cms

process = cms.Process("iptRECOID2")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff")

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
        'rfio:/castor/cern.ch/user/s/safronov/forIsoTracksFromReco.root'
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

process.hltPoolOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('reco_IsoTrNoHLT_TEST.root'),
    outputCommands = cms.untracked.vstring('keep *_IsoProd_*_*')
)

process.AlCaIsoTrTest = cms.Path(process.seqALCARECOHcalCalIsoTrk)
process.HLTPoolOutput = cms.EndPath(process.pts*process.hltPoolOutput)


