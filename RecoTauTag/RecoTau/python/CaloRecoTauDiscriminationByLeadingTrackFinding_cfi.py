import FWCore.ParameterSet.Config as cms

caloRecoTauDiscriminationByLeadingTrackFinding = cms.EDFilter("CaloRecoTauDiscriminationByLeadingTrackFinding",
    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),
)

