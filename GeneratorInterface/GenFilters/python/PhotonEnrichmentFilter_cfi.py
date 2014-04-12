import FWCore.ParameterSet.Config as cms

PhotonEnrichmentFilter = cms.EDFilter("PhotonEnrichmentFilter",
									  Debug = cms.bool(False),
									  #Report = cms.bool(False),
									  ClusterConeSize = cms.double(0.085),
									  EMSeedThreshold = cms.double(8.0),
									  PionSeedThreshold = cms.double(15.0),
									  GenParticleThreshold = cms.double(1.0),
									  SecondarySeedThreshold = cms.double(1.0),
									  IsoConeSize = cms.double(0.4),
									  IsolationCutOff = cms.double(50.0),
									  
									  ClusterEtThreshold = cms.double(20.0),
									  ClusterEtRatio = cms.double(0.50),
									  CaloIsoEtRatio = cms.double(0.60),
									  TrackIsoEtRatio = cms.double(0.40),
									  ClusterTrackEtRatio = cms.double(0.55),
									  
									  MaxClusterCharge = cms.int32(3),
									  ChargedParticleThreshold = cms.int32(4),
									  ClusterNonSeedThreshold = cms.int32(5),
									  ClusterSeedThreshold = cms.int32(9),
									  NumPhotons = cms.int32(1)
									  )
