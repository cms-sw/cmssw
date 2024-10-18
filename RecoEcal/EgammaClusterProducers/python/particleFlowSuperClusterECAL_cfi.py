import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALMustache_cfi import particleFlowSuperClusterECALMustache as _particleFlowSuperClusterECALMustache
# define the default ECAL clustering (Mustache or Box or DeepSC)
particleFlowSuperClusterECAL = _particleFlowSuperClusterECALMustache.clone()

from Configuration.ProcessModifiers.ecal_deepsc_cff import ecal_deepsc
_particleFlowSuperClusterECALDeepSC = _particleFlowSuperClusterECALMustache.clone(
    ClusteringType = "DeepSC",
    deepSuperClusterConfig = cms.PSet(
        modelFiles =  cms.vstring("RecoEcal/EgammaClusterProducers/data/DeepSCModels/models_pfthres/model_noise235fb_thresUL18/model_smallpadding.pb",
                                  "RecoEcal/EgammaClusterProducers/data/DeepSCModels/models_pfthres/model_noise235fb_thresUL18/model_largepadding.pb"),
        configFileClusterFeatures =  cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/models_pfthres/model_noise235fb_thresUL18/scaler_config_cls_norm.txt"),
        configFileWindowFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/models_pfthres/model_noise235fb_thresUL18/scaler_config_wind_norm.txt"),
        configFileHitsFeatures =  cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/models_pfthres/model_noise235fb_thresUL18/config_hits_inputs.txt"),
        collectionStrategy = cms.string("Cascade"),
        maxNClusters = cms.vuint32(15,60),
        maxNRechits = cms.vuint32(20,60),
        nClusterFeatures = cms.uint32(17),
        nWindowFeatures = cms.uint32(6)
    )
)
ecal_deepsc.toReplaceWith(particleFlowSuperClusterECAL, _particleFlowSuperClusterECALDeepSC)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(particleFlowSuperClusterECAL, useDynamicDPhiWindow = False,
                                                phiwidth_SuperClusterBarrel = 0.20,
                                                phiwidth_SuperClusterEndcap = 0.20)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowSuperClusterECAL,
                                thresh_SCEt = 1.0,
                                thresh_PFClusterSeedBarrel = 0.5,
                                thresh_PFClusterSeedEndcap = 0.5)

