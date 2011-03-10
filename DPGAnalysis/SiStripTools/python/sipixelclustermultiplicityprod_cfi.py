import FWCore.ParameterSet.Config as cms


spclustermultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                   clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                   wantedSubDets = cms.VPSet(    
                                                          cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel")),
                                                          cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX")),
                                                          cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"))
                                                          )
                                )
