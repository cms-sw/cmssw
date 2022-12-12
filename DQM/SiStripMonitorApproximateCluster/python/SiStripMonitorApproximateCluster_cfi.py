import FWCore.ParameterSet.Config as cms

# SiStripMonitorCluster
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorApproximateCluster = DQMEDAnalyzer("SiStripMonitorApproximateCluster",
                                                 compareClusters = cms.bool(False),
                                                 ApproxClustersProducer = cms.InputTag('hltSiStripClusters2ApproxClusters'),
                                                 ClustersProducer = cms.InputTag('siStripClusters'),
                                                 folder = cms.string('SiStripApproximateClusters'))
