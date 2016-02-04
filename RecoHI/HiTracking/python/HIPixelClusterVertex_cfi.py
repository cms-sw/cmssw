import FWCore.ParameterSet.Config as cms

hiPixelClusterVertex = cms.EDProducer("HIPixelClusterVtxProducer",
                                      pixelRecHits=cms.string("siPixelRecHits"),
                                      minZ=cms.double(-30.0),
                                      maxZ=cms.double(30.05),
                                      zStep=cms.double(0.1)
                                      )


