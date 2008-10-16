import FWCore.ParameterSet.Config as cms

caloRecoTauDiscriminationByLeadingTrackPtCut = cms.EDFilter("CaloRecoTauDiscriminationByLeadingTrackPtCut",
    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),
    MinPtLeadingTrack = cms.double(5.0)
)


