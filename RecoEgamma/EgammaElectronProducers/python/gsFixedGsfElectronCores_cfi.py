import FWCore.ParameterSet.Config as cms

gsFixedGsfElectronCores = cms.EDProducer("GsfElectronCoreGSCrysFixer",
    orgCores=cms.InputTag("gedGsfElectronCores", '', cms.InputTag.skipCurrentProcess()),
    refinedSCs = cms.InputTag('gsFixedRefinedSuperClusters'),
    scs = cms.InputTag('particleFlowSuperClusterECALGSFixed'),
)
