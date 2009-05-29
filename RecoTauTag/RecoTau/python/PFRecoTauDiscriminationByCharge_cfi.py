import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByCharge = cms.EDFilter("PFRecoTauDiscriminationByCharge",
    PFTauProducer                              = cms.InputTag('pfRecoTauProducerHighEfficiency'),
    PTcut                                      = cms.double(7.0),
    ApplySigTkSum                              = cms.bool(True),
    MinHitsLeadTk                              = cms.double(10),
    MinSigPtTkRatio                            = cms.double(0.05)           
)


