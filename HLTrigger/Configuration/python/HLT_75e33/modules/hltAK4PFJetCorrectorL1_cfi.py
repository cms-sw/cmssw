import FWCore.ParameterSet.Config as cms

hltAK4PFJetCorrectorL1 = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllTmp")
)
