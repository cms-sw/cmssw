import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmGedGsfElectronPFIsolation_cfi import *
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolation_cfi import *

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *

pfNoPileUpCandidates = pfAllChargedHadrons.clone()
pfNoPileUpCandidates.pdgId.extend(pfAllNeutralHadronsAndPhotons.pdgId)

egmIsolationSequence = cms.Sequence( pfParticleSelectionSequence + 
                                     pfNoPileUpCandidates + 
                                     egmGedGsfElectronPFNoPileUpIsolation +
                                     egmGedGsfElectronPFPileUpIsolation +
                                     pfClusterIsolationSequence
                                     )
