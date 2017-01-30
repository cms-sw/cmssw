import FWCore.ParameterSet.Config as cms

particleFlowEGammaGSFixed = cms.EDProducer("EGRefinedSCFixer",
    orgRefinedSC = cms.InputTag("particleFlowEGamma", '', cms.InputTag.skipCurrentProcess()),
    orgSC = cms.InputTag("particleFlowSuperClusterECAL", '', cms.InputTag.skipCurrentProcess()),
    fixedSC = cms.InputTag("particleFlowSuperClusterECALGSFixed"),
    fixedPFClusters = cms.InputTag("particleFlowClusterECALGSFixed"),
    throwOnDupECALClustersInEvent = cms.bool(False),
    throwOnDupESClustersInEvent = cms.bool(True)
                                             
)
