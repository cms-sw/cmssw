import FWCore.ParameterSet.Config as cms

jetCollectionForEleHT = cms.EDProducer('JetCollectionForEleHT',
    HltElectronTag = cms.InputTag("triggerFilterObjectWithRefs"),
    SourceJetTag   = cms.InputTag("caloJetCollection"),
    minDeltaR      = cms.double(0.5)
)
