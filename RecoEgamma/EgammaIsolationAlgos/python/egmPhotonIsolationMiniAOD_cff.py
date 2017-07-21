import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions as _IsoConeDefinitions

egmPhotonIsolation = cms.EDProducer( "CITKPFIsolationSumProducer",
                                     srcToIsolate = cms.InputTag("slimmedPhotons"),
                                     srcForIsolationCone = cms.InputTag('packedPFCandidates'),
                                     isolationConeDefinitions = _IsoConeDefinitions
                                     )	

# The sequence defined here contains only one module. This is to keep the structure
# uniform with the AOD case where there are more modules in the analogous sequence.
egmPhotonIsolationMiniAODTask = cms.Task( egmPhotonIsolation )
egmPhotonIsolationMiniAODSequence = cms.Sequence( egmPhotonIsolationMiniAODTask )

