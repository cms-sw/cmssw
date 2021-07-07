import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import pfPileUpIso, pfNoPileUpIso, pfNoPileUpIsoTask
from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions as _IsoConeDefinitions
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import pfNoPileUpCandidates
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducer_cfi as _mod

egmPhotonIsolation = _mod.CITKPFIsolationSumProducer.clone(
                                     srcToIsolate = "gedPhotons",
                                     srcForIsolationCone = 'pfNoPileUpCandidates',
                                     isolationConeDefinitions = _IsoConeDefinitions
                                     )	

egmPhotonIsolationAODTask = cms.Task(pfNoPileUpIsoTask,
                                     pfNoPileUpCandidates,
                                     egmPhotonIsolation)
egmPhotonIsolationAODSequence = cms.Sequence(egmPhotonIsolationAODTask)
