import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.VectorBosonAnalysis.tupelSequence_pp_cff import *

from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import photonMatch
patPhotonSequence = cms.Sequence(photonMatch + patPhotons)

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import electronMatch
patElectronSequence = cms.Sequence(electronMatch + patElectrons)

tupelPatSequence = cms.Sequence(patMuonsWithTriggerSequence + patPhotonSequence + patElectronSequence + tupel)
