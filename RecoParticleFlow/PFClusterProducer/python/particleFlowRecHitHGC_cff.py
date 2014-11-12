from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCEE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCHEF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCHEB_cfi import *

particleFlowRecHitHGCAll = cms.EDProducer(
    "PFRecHitMerger",
    src = cms.VInputTag( cms.InputTag("particleFlowRecHitHGCEE")
                         cms.InputTag("particleFlowRecHitHGCHEF"),
                         cms.InputTag("particleFlowRecHitHGCHEB")
                         )
)

particleFlowRecHitHGC = cms.Sequence( particleFlowRecHitHGCEE  +
                                      particleFlowRecHitHGCHEF +
                                      particleFlowRecHitHGCHEB +
                                      particleFlowRecHitHGCAll )
