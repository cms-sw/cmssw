import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import pfPileUpIso, pfNoPileUpIso, pfNoPileUpIsoSequence
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedHadrons_cfi import pfAllChargedHadrons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadronsAndPhotons_cfi import pfAllNeutralHadronsAndPhotons
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff import IsoConeDefinitions
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import pfNoPileUpCandidates


egmPhotonIsolation = cms.EDProducer( "CITKPFIsolationSumProducer",
                                     srcToIsolate = cms.InputTag("gedPhotons"),
                                     srcForIsolationCone = cms.InputTag('pfNoPileUpCandidates'),
                                     isolationConeDefinitions = IsoConeDefinitions
                                     )	

egmPhotonIsolationAODSequence = cms.Sequence(pfNoPileUpIsoSequence + pfNoPileUpCandidates + egmPhotonIsolation)

