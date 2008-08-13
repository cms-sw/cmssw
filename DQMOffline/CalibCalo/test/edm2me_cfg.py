import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:dqm.root')
)

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.EDMtoMEConverter.Verbosity = 1
process.EDMtoMEConverter.Frequency = 1
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/A/B/C'


