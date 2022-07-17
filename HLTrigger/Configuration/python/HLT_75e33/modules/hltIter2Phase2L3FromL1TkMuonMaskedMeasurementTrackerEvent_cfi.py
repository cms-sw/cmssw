import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonMaskedMeasurementTrackerEvent = cms.EDProducer("MaskedMeasurementTrackerEventProducer",
    OnDemand = cms.bool(False),
    phase2clustersToSkip = cms.InputTag("hltIter2Phase2L3FromL1TkMuonClustersRefRemoval"),
    src = cms.InputTag("MeasurementTrackerEvent")
)
