import FWCore.ParameterSet.Config as cms



from DQMOffline.EGamma.photonAnalyzer_cfi import *
from DQMOffline.EGamma.electronAnalyzerSequence_cff import *

photonAnalysis.OutputMEsInRootFile = cms.bool(False)
photonAnalysis.Verbosity = cms.untracked.int32(0)

gsfElectronAnalysis.OutputMEsInRootFile = cms.bool(False)
gsfElectronAnalysis.Verbosity = cms.untracked.int32(0)

egammaDQMOffline = cms.Sequence(photonAnalysis*electronAnalyzerSequence)
