import FWCore.ParameterSet.Config as cms

process = cms.Process("Mon")
#    service = Tracer {}
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

# Magnetic fiuld: force mag field to be 0.0 tesla
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

# reconstruction sequence for Global Run
process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

# offline raw to digi
process.load("Configuration.GlobalRuns.RawToDigiGR_cff")

# offline DQM
process.load("DQMOffline.Configuration.DQMOffline_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:reco.root')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('DQMTest.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)
process.allPath = cms.Path(process.DQMOffline*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''

