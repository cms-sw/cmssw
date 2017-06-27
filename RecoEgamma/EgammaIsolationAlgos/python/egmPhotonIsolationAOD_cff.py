import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import pfPileUpIso, pfNoPileUpIso, pfNoPileUpIsoTask
from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions as _IsoConeDefinitions
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import pfNoPileUpCandidates


egmPhotonIsolation = cms.EDProducer( "CITKPFIsolationSumProducer",
                                     srcToIsolate = cms.InputTag("gedPhotons"),
                                     srcForIsolationCone = cms.InputTag('pfNoPileUpCandidates'),
                                     isolationConeDefinitions = _IsoConeDefinitions
                                     )	

egmPhotonIsolationAODTask = cms.Task(pfNoPileUpIsoTask,
                                     pfNoPileUpCandidates,
                                     egmPhotonIsolation)
egmPhotonIsolationAODSequence = cms.Sequence(egmPhotonIsolationAODTask)
