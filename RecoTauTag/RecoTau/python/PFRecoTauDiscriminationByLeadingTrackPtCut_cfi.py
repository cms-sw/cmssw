import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByLeadingTrackPtCut = cms.EDFilter("PFRecoTauDiscriminationByLeadingTrackPtCut",
    PFTauProducer = cms.string('pfRecoTauProducer'),
    MinPtLeadingTrack = cms.double(5.0)
)


