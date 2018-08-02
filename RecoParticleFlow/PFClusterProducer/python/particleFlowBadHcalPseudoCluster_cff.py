from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cfi import *

# OFF by default, turned on via modifier for the moment
particleFlowBadHcalPseudoCluster.enable = cms.bool(False)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(particleFlowBadHcalPseudoCluster, thresholdHE = 4)

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(particleFlowBadHcalPseudoCluster, thresholdHB = 4, thresholdHE = 4)
