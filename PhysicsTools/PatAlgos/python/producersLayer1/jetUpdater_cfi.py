import FWCore.ParameterSet.Config as cms

patJetsUpdated = cms.EDProducer("PATJetUpdater",
    # input
    jetSource = cms.InputTag("slimmedJets"),
    # jet energy corrections
    addJetCorrFactors    = cms.bool(True),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsUpdated") ),
)


