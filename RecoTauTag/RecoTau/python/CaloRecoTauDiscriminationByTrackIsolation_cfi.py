import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrackCalo

caloRecoTauDiscriminationByTrackIsolation = cms.EDProducer("CaloRecoTauDiscriminationByIsolation",

    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),

    Prediscriminants = requireLeadTrackCalo,

    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    TrackerIsolAnnulus_maximumOccupancy   = cms.uint32(0),

    ApplyDiscriminationByECALIsolation    = cms.bool(False),
    ECALisolAnnulus_maximumSumEtCut       = cms.double(1.5)
)


