import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

pfRecoTauDiscriminationByCharge = cms.EDProducer("PFRecoTauDiscriminationByCharge",
    # tau collection to discriminate
    PFTauProducer                              = cms.InputTag('pfRecoTauProducerHighEfficiency'),

    # no prediscriminants needed
    Prediscriminants = noPrediscriminants,

    # If true, additionally fail taus that do not have 1 or 3 tracks in the signal cone
    ApplyOneOrThreeProngCut                    = cms.bool(False),

    # Requirement on charge of tau (signal contents only)
    AbsChargeReq                               = cms.uint32(1)
)


