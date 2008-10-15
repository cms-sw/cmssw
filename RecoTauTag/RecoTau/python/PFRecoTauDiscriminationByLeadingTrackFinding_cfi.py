import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByLeadingTrackFinding = cms.EDProducer("PFRecoTauDiscriminationByLeadingTrackFinding",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
)

