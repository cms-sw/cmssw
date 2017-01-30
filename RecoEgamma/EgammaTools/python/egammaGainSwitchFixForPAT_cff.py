import FWCore.ParameterSet.Config as cms 

from RecoEgamma.EgammaTools.egammaGainSwitchFix_cff import *

# rename the products that stay in AOD event content
particleFlowEGammaGSFixed = gsFixedRefinedSuperClusters.clone(
    fixedSC = 'particleFlowSuperClusterECALGSFixed',
    fixedPFClusters = 'particleFlowClusterECALGSFixed'
)
gedGsfElectronCoresGSFixed = gsFixedGsfElectronCores.clone(
    refinedSCs = 'particleFlowEGammaGSFixed',
    scs = 'particleFlowSuperClusterECALGSFixed'
)
gedGsfElectronsGSFixed = gsFixedGsfElectrons.clone(
    newCores = 'gedGsfElectronCoresGSFixed'
)
gedPhotonCoreGSFixed = gsFixedGEDPhotonCores.clone(
    refinedSCs = 'particleFlowEGammaGSFixed',
    scs = 'particleFlowSuperClusterECALGSFixed',
    conversions = 'allConversions',
    singleconversions = 'particleFlowEGamma'
)
gedPhotonsGSFixed = gsFixedGEDPhotons.clone(
    newCores = 'gedPhotonCoreGSFixed'
)

gsFixedParticleFlowSuperClustering = cms.Sequence(
    particleFlowRecHitECALGSFixed*
    particleFlowRecHitPSGSFixed*
    particleFlowClusterPSGSFixed*
    particleFlowClusterECALUncorrectedGSFixed*
    particleFlowClusterECALGSFixed*
    particleFlowSuperClusterECALGSFixed
)

egammaGainSwitchFixSequence = cms.Sequence(
    egammaGainSwitchLocalFixSequence*
    gsFixedParticleFlowSuperClustering*
    particleFlowEGammaGSFixed*
    gedGsfElectronCoresGSFixed*
    gedGsfElectronsGSFixed*
    gedPhotonCoreGSFixed*
    gedPhotonsGSFixed
)
