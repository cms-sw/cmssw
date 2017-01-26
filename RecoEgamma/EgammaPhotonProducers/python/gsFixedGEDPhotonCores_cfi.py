import FWCore.ParameterSet.Config as cms

gsFixedGEDPhotonCores = cms.EDProducer("GEDPhotonCoreGSCrysFixer",
    photonCores = cms.InputTag("gedPhotonCore", '', cms.InputTag.skipCurrentProcess()),
    refinedSCs = cms.InputTag('gsFixedRefinedSuperClusters'),
    scs = cms.InputTag('particleFlowSuperClusterECALGSFixed'),
    conversions = cms.InputTag('gsFixedConversions')
)
