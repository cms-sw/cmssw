import FWCore.ParameterSet.Config as cms

hltAK8PFPuppiJetCorrectorL1 = cms.EDProducer("L1FastjetCorrectorProducer",
    #algorithm = cms.string('AK8PFPuppiHLT'),
    algorithm = cms.string('AK8PFPuppi'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllTmp")
)
