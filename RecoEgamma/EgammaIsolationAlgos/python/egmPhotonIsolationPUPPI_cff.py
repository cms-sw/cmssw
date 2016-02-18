import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff import IsoConeDefinitions

egmPhotonIsolationAODPUPPI = cms.EDProducer( "CITKPFIsolationSumProducerForPUPPI",
			  srcToIsolate = cms.InputTag("gedPhotons"),
			  srcForIsolationCone = cms.InputTag('particleFlow'),
                          puppiValueMap = cms.InputTag('puppi'),
			  isolationConeDefinitions = IsoConeDefinitions
)

egmPhotonIsolationMiniAODPUPPI = egmPhotonIsolationAODPUPPI.clone()
egmPhotonIsolationMiniAODPUPPI.srcForIsolationCone = cms.InputTag("packedPFCandidates")
egmPhotonIsolationMiniAODPUPPI.srcToIsolate = cms.InputTag("slimmedPhotons")
egmPhotonIsolationMiniAODPUPPI.puppiValueMap = cms.InputTag('')
