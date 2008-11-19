import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByLeadingTrackPtCut = cms.EDFilter("PFRecoTauDiscriminationByLeadingTrackPtCut",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
    MinPtLeadingPion = cms.double(5.0)
)


