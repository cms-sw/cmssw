from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cfi import *

# ON by default, turned off via modifier
from Configuration.Eras.Modifier_pf_badHcalMitigationOff_cff import pf_badHcalMitigationOff
pf_badHcalMitigationOff.toModify(particleFlowBadHcalPseudoCluster, enable = False)


