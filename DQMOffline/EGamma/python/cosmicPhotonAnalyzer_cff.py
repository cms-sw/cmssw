import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.cosmicPhotonAnalyzer_cfi import *
cosmicPhotonAnalysis.OutputMEsInRootFile = cms.bool(False)
cosmicPhotonAnalysis.Verbosity = cms.untracked.int32(0)

egammaCosmicPhotonMonitors = cms.Sequence(cosmicPhotonAnalysis)
