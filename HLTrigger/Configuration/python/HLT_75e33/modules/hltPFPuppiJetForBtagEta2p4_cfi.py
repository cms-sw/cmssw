import FWCore.ParameterSet.Config as cms

hltPFPuppiJetForBtagEta2p4 = cms.EDProducer("HLTPFJetCollectionProducer",
    HLTObject = cms.InputTag("hltPFPuppiJetForBtagSelectorEta2p4"),
    TriggerTypes = cms.vint32(86)
)
