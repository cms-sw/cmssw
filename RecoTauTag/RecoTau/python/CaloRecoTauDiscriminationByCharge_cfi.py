import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

caloRecoTauDiscriminationByCharge = cms.EDProducer("CaloRecoTauDiscriminationByCharge",
    # tau collection to discriminate
    CaloTauProducer                              = cms.InputTag('caloRecoTauProducerHighEfficiency'),

    # no prediscriminants needed
    Prediscriminants = noPrediscriminants,

    # If true, additionally fail taus that do not have 1 or 3 tracks in the signal cone
    ApplyOneOrThreeProngCut                    = cms.bool(False),

    # Requirement on charge of tau (signal contents only)
    AbsChargeReq                               = cms.uint32(1)
)
