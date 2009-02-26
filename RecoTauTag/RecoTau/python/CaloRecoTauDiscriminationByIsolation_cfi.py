import FWCore.ParameterSet.Config as cms

caloRecoTauDiscriminationByIsolation = cms.EDFilter("CaloRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0)
)


