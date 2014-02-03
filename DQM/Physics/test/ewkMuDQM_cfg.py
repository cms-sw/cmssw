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
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/00EC507F-CD01-E111-8EA1-BCAEC5329724.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/02312EAD-F801-E111-8370-001D09F251D1.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/0257B47B-0002-E111-A397-003048F1183E.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/041C6C9D-F601-E111-A1C8-001D09F251FE.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/08D80B70-0502-E111-81B1-002481E0D958.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/0CB1E6B5-1F03-E111-924B-003048F118DE.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/102E699B-AB01-E111-8D71-BCAEC53296FC.root",
 "/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/179/889/10C174C9-D401-E111-AE6D-00237DDC5BBC.root"
)
)


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            threshold = cms.untracked.string('DEBUG')
           #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkMuDQM+process.dqmSaver)

