import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkMuDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/Muon')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
  
"rfio:/castor/cern.ch/cms/store/data/Run2010B/Mu/RECO/PromptReco-v2/000/149/442/B0642007-47E6-DF11-ACC0-0030487CD16E.root"
)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('DEBUG')
            #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkMuDQM+process.dqmSaver)

