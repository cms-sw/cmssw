import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByLeadingPionPtCut = cms.EDFilter("PFRecoTauDiscriminationByLeadingPionPtCut",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
    MinPtLeadingPion = cms.double(5.0)
)


