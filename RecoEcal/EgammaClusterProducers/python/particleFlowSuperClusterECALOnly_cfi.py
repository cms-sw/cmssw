import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALMustache_cfi import particleFlowSuperClusterECALMustache as _particleFlowSuperClusterECALMustache

# "Mustache" clustering
particleFlowSuperClusterECALOnly = _particleFlowSuperClusterECALMustache.clone(
    # ECAL-only (no primary vertices) regression setup
    regressionConfig = dict(
        isHLT = True,
        eRecHitThreshold = 1.,
        regressionKeyEB  = 'pfscecal_EBCorrection_online',
        uncertaintyKeyEB = 'pfscecal_EBUncertainty_online',
        regressionKeyEE  = 'pfscecal_EECorrection_online',
        uncertaintyKeyEE = 'pfscecal_EEUncertainty_online',
        vertexCollection = '',
    ),
    # ECAL-only (no primary vertices) thresholds
    thresh_PFClusterBarrel = 0.5,
    thresh_PFClusterEndcap = 0.5,
    thresh_PFClusterES     = 0.5,
)
