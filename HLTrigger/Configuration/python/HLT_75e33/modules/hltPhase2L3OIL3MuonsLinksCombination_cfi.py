import FWCore.ParameterSet.Config as cms

hltPhase2L3OIL3MuonsLinksCombination = cms.EDProducer("L3TrackLinksCombiner",
    labels = cms.VInputTag("hltL3MuonsPhase2L3OI")
)
