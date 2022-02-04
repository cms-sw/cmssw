import FWCore.ParameterSet.Config as cms

hltPhase2L3OIL3Muons = cms.EDProducer("L3TrackCombiner",
    labels = cms.VInputTag("hltL3MuonsPhase2L3OI")
)
