import FWCore.ParameterSet.Config as cms

hltPFPuppiJetForBtagEta4p0 = cms.EDProducer("HLTPFJetCollectionProducer",
    HLTObject = cms.InputTag("hltPFPuppiJetForBtagSelectorEta4p0"),
    TriggerTypes = cms.vint32(86)
)
