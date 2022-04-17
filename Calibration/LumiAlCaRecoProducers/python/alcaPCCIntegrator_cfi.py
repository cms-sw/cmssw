import FWCore.ParameterSet.Config as cms 

alcaPCCIntegrator = cms.EDProducer("AlcaPCCIntegrator",
    AlcaPCCIntegratorParameters = cms.PSet(
        inputPccLabel = cms.string("hltAlcaPixelClusterCounts"),
        trigstring = cms.untracked.string("alcaPCCEvent"),
        ProdInst = cms.string("alcaPCCRandom")
    )
)
