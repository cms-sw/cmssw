import FWCore.ParameterSet.Config as cms

gsFixedConversions = cms.EDProducer('ConversionGSCrysFixer',
    conversions = cms.InputTag('allConversions', '', cms.InputTag.skipCurrentProcess()),
    superClusters = cms.InputTag('particleFlowSuperClusterECALGSFixed'),
    scMaps = cms.InputTag('gsFixedRefinedSuperClusters')
)
