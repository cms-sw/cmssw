import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonMaskedMeasurementTrackerEvent = cms.EDProducer("MaskedMeasurementTrackerEventProducer",
    phase2clustersToSkip = cms.InputTag("hltIter2Phase2L3FromL1TkMuonClustersRefRemoval"),
    src = cms.InputTag("hltMeasurementTrackerEvent")
)
