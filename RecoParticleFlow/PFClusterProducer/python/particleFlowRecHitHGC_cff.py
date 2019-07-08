from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cfi import *

particleFlowRecHitHGCTask = cms.Task( particleFlowRecHitHGC )
particleFlowRecHitHGCSeq = cms.Sequence( particleFlowRecHitHGCTask )
