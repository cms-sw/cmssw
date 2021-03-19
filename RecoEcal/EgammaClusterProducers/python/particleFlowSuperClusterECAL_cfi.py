import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALMustache_cfi import particleFlowSuperClusterECALMustache as _particleFlowSuperClusterECALMustache

# define the default ECAL clustering (Mustache or Box)
particleFlowSuperClusterECAL = _particleFlowSuperClusterECALMustache.clone()

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(particleFlowSuperClusterECAL, useDynamicDPhiWindow = False)
pp_on_AA.toModify(particleFlowSuperClusterECAL, phiwidth_SuperClusterBarrel = 0.20)
pp_on_AA.toModify(particleFlowSuperClusterECAL, phiwidth_SuperClusterEndcap = 0.20)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowSuperClusterECAL,
                                thresh_SCEt = 1.0,
                                thresh_PFClusterSeedBarrel = 0.5,
                                thresh_PFClusterSeedEndcap = 0.5)
