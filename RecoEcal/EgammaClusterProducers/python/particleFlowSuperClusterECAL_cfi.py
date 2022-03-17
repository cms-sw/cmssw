import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALMustache_cfi import particleFlowSuperClusterECALMustache as _particleFlowSuperClusterECALMustache
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALDeepSC_cfi import particleFlowSuperClusterECALDeepSC as _particleFlowSuperClusterECALDeepSC
# define the default ECAL clustering (Mustache or Box or DeepSC)
particleFlowSuperClusterECAL = _particleFlowSuperClusterECALMustache.clone()

from Configuration.ProcessModifiers.ecal_deepsc_cff import ecal_deepsc
ecal_deepsc.toReplaceWith(particleFlowSuperClusterECAL, _particleFlowSuperClusterECALDeepSC.clone())

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(particleFlowSuperClusterECAL, useDynamicDPhiWindow = False,
                                                phiwidth_SuperClusterBarrel = 0.20,
                                                phiwidth_SuperClusterEndcap = 0.20)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowSuperClusterECAL,
                                thresh_SCEt = 1.0,
                                thresh_PFClusterSeedBarrel = 0.5,
                                thresh_PFClusterSeedEndcap = 0.5)

