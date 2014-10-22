import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.TCMET_cfi import *

tcMetWithPFclusters = tcMet.clone(
        PFClustersHFEM = cms.InputTag('particleFlowClusterHF'),
        PFClustersHFHAD = cms.InputTag('particleFlowClusterHF')
    )

tcMetWithPFclusters.usePFClusters = cms.bool(True)

tcMetSequence = cms.Sequence(tcMet + tcMetWithPFclusters)
