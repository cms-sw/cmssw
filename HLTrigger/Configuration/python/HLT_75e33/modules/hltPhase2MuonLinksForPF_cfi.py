import FWCore.ParameterSet.Config as cms

# Re-point the tracker track of each trigger-muon link into hltGeneralTracks
hltPhase2MuonLinksForPF = cms.EDProducer("MuonLinksProducerForHLT",
    LinkCollection = cms.InputTag("hltPhase2L3MuonLinks"),
    InclusiveTrackerTrackCollection = cms.InputTag("hltGeneralTracks"),
    ptMin = cms.double(2.5),
    pMin = cms.double(2.5),
    shareHitFraction = cms.double(0.8)
)
