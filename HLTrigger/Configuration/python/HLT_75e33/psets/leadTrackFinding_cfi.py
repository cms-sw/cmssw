import FWCore.ParameterSet.Config as cms

leadTrackFinding = cms.PSet(
    Producer = cms.InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"),
    cut = cms.double(0.5)
)