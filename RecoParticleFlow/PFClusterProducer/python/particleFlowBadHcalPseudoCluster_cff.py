from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cfi import *

# OFF by default, turned on via modifier
from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify(particleFlowBadHcalPseudoCluster, enable = True)

