import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPileUpDepMonitor = DQMEDAnalyzer('EcalPileUpDepMonitor',
			   VertexCollection = cms.InputTag("offlinePrimaryVertices"),
                           basicClusterCollection = cms.InputTag("particleFlowClusterECAL"),
                           superClusterCollection_EE = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"),
			   superClusterCollection_EB = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"),
			   RecHitCollection_EB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
			   RecHitCollection_EE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
			   EleTag              = cms.InputTag("gedGsfElectrons")
                      )

