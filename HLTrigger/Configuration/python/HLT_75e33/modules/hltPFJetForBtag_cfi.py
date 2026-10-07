import FWCore.ParameterSet.Config as cms

hltPFJetForBtag = cms.EDProducer( "HLTPFJetCollectionProducer",
    HLTObject = cms.InputTag( "hltPFJetForBtagSelector" ),
    TriggerTypes = cms.vint32( 86 )
)
