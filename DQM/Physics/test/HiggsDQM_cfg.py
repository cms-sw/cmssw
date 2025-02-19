import FWCore.ParameterSet.Config as cms

process = cms.Process("HiggsDQM")
process.load("DQM.Physics.HiggsDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/Higgs')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
             '/store/data/Run2010A/Mu/RAW/v1/000/139/370/EE40C530-7787-DF11-A515-001D09F231B0.root',
             '/store/data/Run2010A/Mu/RAW/v1/000/139/370/D4DEBC75-5C87-DF11-96E4-003048F118DE.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.HiggsDQM+process.dqmSaver)

