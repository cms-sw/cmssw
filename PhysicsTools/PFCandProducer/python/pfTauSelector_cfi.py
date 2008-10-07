import FWCore.ParameterSet.Config as cms


pfTaus = cms.EDProducer("PFTauSelector",
    src = cms.InputTag("pfRecoTauProducer"),
    discriminator = cms.InputTag("pfRecoTauDiscriminationByIsolation")
)
