import FWCore.ParameterSet.Config as cms

hiPixelClusterVertex = cms.EDProducer("HIPixelClusterVtxProducer",
                                      pixelRecHits=cms.untracked.string("siPixelRecHits"),
                                      minZ=cms.untracked.double(-20.0),
                                      maxZ=cms.untracked.double(20.05),
                                      zStep=cms.untracked.double(0.1)
                                      )


