import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.VectorBosonAnalysis.tupelSequence_pp_cff import *

from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import photonMatch
patPhotonSequence.remove([patPhotons])
patPhotonSequence.append([photonMatch, patPhotons])

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import electronMatch
patElectronSequence.remove([patElectrons])
patElectronSequence.append([electronMatch, patElectrons])
