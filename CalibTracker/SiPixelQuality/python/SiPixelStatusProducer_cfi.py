import FWCore.ParameterSet.Config as cms

siPixelStatusProducer = cms.EDProducer("SiPixelStatusProducer",
    SiPixelStatusProducerParameters = cms.PSet(
        badPixelFEDChannelCollections = cms.VInputTag(cms.InputTag('siPixelDigis')),
        pixelClusterLabel = cms.untracked.InputTag("siPixelClusters::RECO"),
    )
)

