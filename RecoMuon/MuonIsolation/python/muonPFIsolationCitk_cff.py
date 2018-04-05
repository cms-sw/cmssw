import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolation.muonPFIsolationCitk_cfi import *

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *

pfNoPileUpCandidates = pfAllChargedHadrons.clone()
pfNoPileUpCandidates.pdgId.extend(pfAllNeutralHadronsAndPhotons.pdgId)

muonIsolationTask = cms.Task(pfParticleSelectionTask,
                             pfNoPileUpCandidates,
                             muonPFNoPileUpIsolation,
                             muonPFPileUpIsolation)
muonIsolationSequence = cms.Sequence(muonIsolationTask)
