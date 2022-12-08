import FWCore.ParameterSet.Config as cms

# SiStripMonitorCluster
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorApproximateCluster = DQMEDAnalyzer("SiStripMonitorApproximateCluster",
                                                 ClusterProducerStrip = cms.InputTag('hltSiStripClusters2ApproxClusters'),
                                                 folder = cms.string('SiStripApproximateClusters'))
