import FWCore.ParameterSet.Config as cms 

alcaPCCProducer = cms.EDProducer("AlcaPCCProducer",
    pixelClusterLabel = cms.InputTag("siPixelClustersForLumi"),
    trigstring = cms.untracked.string("alcaPCC") 
)
