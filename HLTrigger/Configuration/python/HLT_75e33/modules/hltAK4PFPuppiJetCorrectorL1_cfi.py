import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrectorL1 = cms.EDProducer("L1FastjetCorrectorProducer",
    #algorithm = cms.string('AK4PFPuppiHLT'),
    algorithm = cms.string('AK4PFPuppi'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllTmp")
)
