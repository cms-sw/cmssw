import FWCore.ParameterSet.Config as cms

hltEgammaTriggerFilterObjectWrapper= cms.EDFilter( "HLTEgammaTriggerFilterObjectWrapper",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    doIsolated = cms.bool( False ),
    saveTags = cms.bool( False )
)

