import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByLeadingTrackPtCut = cms.EDFilter("PFRecoTauDiscriminationByLeadingPionPtCut",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
    MinPtLeadingPion = cms.double(5.0)
)


