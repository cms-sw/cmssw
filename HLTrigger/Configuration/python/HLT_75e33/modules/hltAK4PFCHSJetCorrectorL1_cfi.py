import FWCore.ParameterSet.Config as cms

hltAK4PFCHSJetCorrectorL1 = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllTmp")
)
# foo bar baz
# Z0tc74NtBqntI
# L9uj9P0Brr79O
