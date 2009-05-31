import FWCore.ParameterSet.Config as cms

pfRecoTauLogicalDiscriminator = cms.EDFilter("PFTauDiscriminatorLogicalAndProducer",
    TauCollection                              = cms.InputTag('pfRecoTauProducerHighEfficiency'),
    And                                        = cms.bool(True),
    Or                                         = cms.bool(False),
    TauDiscriminators                          =cms.VInputTag(
    'pfRecoTauDiscriminationByCharge'
    )                                        
)
