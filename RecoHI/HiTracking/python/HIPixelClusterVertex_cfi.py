import FWCore.ParameterSet.Config as cms

hiPixelClusterVertex = cms.EDProducer("HIPixelClusterVtxProducer",
                                      pixelRecHits=cms.string("siPixelRecHits"),
                                      minZ=cms.double(-20.0),
                                      maxZ=cms.double(20.05),
                                      zStep=cms.double(0.1)
                                      )


