from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cfi import *

# OFF by default, turned on via modifier
from Configuration.Eras.Modifier_PF_badHcalMitigation_cff import PF_badHcalMitigation
PF_badHcalMitigation.toModify(particleFlowBadHcalPseudoCluster, enable = True)

