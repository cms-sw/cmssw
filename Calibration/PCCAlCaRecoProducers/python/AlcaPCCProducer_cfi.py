import FWCore.ParameterSet.Config as cms 

alcaPCCProducer = cms.EDProducer("AlcaPCCProducer",
    AlcaPCCProducerParameters = cms.PSet(
        pixelClusterLabel = cms.InputTag("siPixelClustersForLumi"),
        #Mod factor to count lumi and the string to specify output 
        trigstring = cms.untracked.string("alcaPCC") 
    ),
)
