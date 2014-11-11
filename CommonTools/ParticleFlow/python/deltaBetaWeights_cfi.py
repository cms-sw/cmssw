import FWCore.ParameterSet.Config as cms


pfWeightedPhotons = cms.EDProducer('DeltaBetaWeights',
                                   src = cms.InputTag('pfAllPhotons'),
                                   chargedFromPV = cms.InputTag("pfAllChargedParticles"),
                                   chargedFromPU = cms.InputTag("pfPileUpAllChargedParticles")
)
pfWeightedNeutralHadrons = cms.EDProducer('DeltaBetaWeights',
                                   src = cms.InputTag('pfAllNeutralHadrons'),
                                   chargedFromPV = cms.InputTag("pfAllChargedParticles"),
                                   chargedFromPU = cms.InputTag("pfPileUpAllChargedParticles")
)



