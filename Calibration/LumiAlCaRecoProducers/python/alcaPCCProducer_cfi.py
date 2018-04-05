import FWCore.ParameterSet.Config as cms 

alcaPCCProducer = cms.EDProducer("AlcaPCCProducer",
    AlcaPCCProducerParameters = cms.PSet(
        pixelClusterLabel = cms.InputTag("siPixelClustersForLumi"),
        trigstring = cms.untracked.string("alcaPCC") 
    ),
)
