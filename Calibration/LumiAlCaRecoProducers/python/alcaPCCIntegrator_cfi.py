import FWCore.ParameterSet.Config as cms 

alcaPCCIntegrator = cms.EDProducer("AlcaPCCIntegrator",
    AlcaPCCIntegratorParameters = cms.PSet(
        inputPccLabel = cms.InputTag("hltAlcaPixelClusterCounts","alcaPCCEvent"),
        ProdInst = cms.string("alcaPCCRandom")
    )
)
# foo bar baz
# r2cPyfOsxA2Lc
# MB3weSXWQGp51
