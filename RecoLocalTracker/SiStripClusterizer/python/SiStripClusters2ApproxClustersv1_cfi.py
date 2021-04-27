import FWCore.ParameterSet.Config as cms

SiStripClusters2ApproxClustersv1 = cms.EDProducer("SiStripClusters2ApproxClustersv1",
	inputClusters = cms.InputTag("siStripClusters")
)

