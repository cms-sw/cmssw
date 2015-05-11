import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolation.muonPFIsolationCitk_cfi import *

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *

pfNoPileUpCandidates = pfAllChargedHadrons.clone()
pfNoPileUpCandidates.pdgId.extend(pfAllNeutralHadronsAndPhotons.pdgId)

muonIsolationSequence = cms.Sequence( pfParticleSelectionSequence +
                                     pfNoPileUpCandidates +
                                     muonPFNoPileUpIsolation +
                                     muonPFPileUpIsolation )

