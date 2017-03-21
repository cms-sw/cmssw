import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions

egmPhotonIsolationMiniAOD = cms.EDProducer( "CITKPFIsolationSumProducer",
			  srcToIsolate = cms.InputTag("slimmedPhotons"),
			  srcForIsolationCone = cms.InputTag('packedPFCandidates'),
			  isolationConeDefinitions = IsoConeDefinitions
  )	
