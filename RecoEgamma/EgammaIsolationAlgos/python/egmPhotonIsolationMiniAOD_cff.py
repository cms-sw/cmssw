import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions as _IsoConeDefinitions
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducer_cfi as _mod

egmPhotonIsolation = _mod.CITKPFIsolationSumProducer.clone(
                                     srcToIsolate = "slimmedPhotons",
                                     srcForIsolationCone = 'packedPFCandidates',
                                     isolationConeDefinitions = _IsoConeDefinitions
                                     )	

# The sequence defined here contains only one module. This is to keep the structure
# uniform with the AOD case where there are more modules in the analogous sequence.
egmPhotonIsolationMiniAODTask = cms.Task( egmPhotonIsolation )
egmPhotonIsolationMiniAODSequence = cms.Sequence( egmPhotonIsolationMiniAODTask )

