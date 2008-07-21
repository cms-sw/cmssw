import FWCore.ParameterSet.Config as cms

TauRefCombiner = cms.EDFilter("HLTTauRefCombiner",
    InputCollections = cms.VInputTag(cms.InputTag("TauMCProducer","Taus"), cms.InputTag("TauRefProducer","PFTaus")),
    MatchDeltaR = cms.double(0.3),
    OutputCollection = cms.string('GoodTaus')
)



