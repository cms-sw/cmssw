from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cfi import *

# OFF by default, turned on via modifier for the moment
particleFlowBadHcalPseudoCluster.enable = False

from Configuration.Eras.Modifier_PF_badHcalMitigation_cff import PF_badHcalMitigation
PF_badHcalMitigation.toModify(particleFlowBadHcalPseudoCluster, enable = True)

