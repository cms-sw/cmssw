import FWCore.ParameterSet.Config as cms

# module to produce jet correction factors associated in a valuemap
jetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
    jetSource = cms.InputTag("iterativeCone5CaloJets"),
    # Basic JES correction, applied in PAT Layer 1 to pat::Jets
    defaultJetCorrector = cms.string('MCJetCorrectorIcone5'),
    # L5 Flavour corrections, on top of 'defaultJetCorrector'
    udsJetCorrector   = cms.string('L5FlavorJetCorrectorUds'),
    gluonJetCorrector = cms.string('L5FlavorJetCorrectorGluon'),
    cJetCorrector     = cms.string('L5FlavorJetCorrectorC'),
    bJetCorrector     = cms.string('L5FlavorJetCorrectorB'),
)


