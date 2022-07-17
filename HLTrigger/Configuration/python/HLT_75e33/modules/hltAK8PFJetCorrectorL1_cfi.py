import FWCore.ParameterSet.Config as cms

hltAK8PFJetCorrectorL1 = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllTmp")
)
