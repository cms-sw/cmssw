import FWCore.ParameterSet.Config as cms

pfSelectedElectrons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("pfElectronsFromVertex"),
    cut = cms.string("pt>5 && gsfTrackRef.isNonnull && gsfTrackRef.trackerExpectedHitsInner.numberOfHits<2")
)




