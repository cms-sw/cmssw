import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    # untracked string ClusterCollLabel   = "siPixelClusters"
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    #TanLorentzAnglePerTesla = cms.double(0.106),
    #Alpha2Order = cms.bool(True),
    #speed = cms.int32(0)

)


