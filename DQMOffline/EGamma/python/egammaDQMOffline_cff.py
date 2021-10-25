import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import *
from DQMOffline.EGamma.zmumugammaAnalyzer_cfi import *
from DQMOffline.EGamma.piZeroAnalyzer_cfi import *
from DQMOffline.EGamma.electronAnalyzerSequence_cff import *

import DQMOffline.EGamma.zmumugammaAnalyzer_cfi
import DQMOffline.EGamma.photonAnalyzer_cfi

photonAnalysis.OutputMEsInRootFile = cms.bool(False)
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.standAlone = cms.bool(False)


stdPhotonAnalysis = DQMOffline.EGamma.photonAnalyzer_cfi.photonAnalysis.clone(
  ComponentName = cms.string('stdPhotonAnalysis'),
  analyzerName = cms.string('stdPhotonAnalyzer'),
  phoProducer = cms.InputTag('photons'),
  OutputMEsInRootFile = cms.bool(False),
  Verbosity = cms.untracked.int32(0),
  standAlone = cms.bool(False),
)

piZeroAnalysis.OutputMEsInRootFile = cms.bool(False)
piZeroAnalysis.Verbosity = cms.untracked.int32(0)
piZeroAnalysis.standAlone = cms.bool(False)


zmumugammaOldAnalysis = DQMOffline.EGamma.zmumugammaAnalyzer_cfi.zmumugammaAnalysis.clone()
zmumugammaOldAnalysis.ComponentName = cms.string('zmumugammaOldAnalysis')
zmumugammaOldAnalysis.analyzerName = cms.string('zmumugammaOldValidation')
zmumugammaOldAnalysis.phoProducer = cms.InputTag('photons')

# HGCal customizations
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
stdPhotonAnalysisHGCal = stdPhotonAnalysis.clone(
  ComponentName = 'stdPhotonAnalyzerHGCal',
  analyzerName = 'stdPhotonAnalyzerHGCal',
  phoProducer = 'photonsHGC',
  isolationStrength = 2,
  etaMin = -3.0,
  etaMax = 3.0,
  maxPhoEta = 3.0,
)

egammaDQMOffline = cms.Sequence(photonAnalysis*stdPhotonAnalysis*zmumugammaOldAnalysis*zmumugammaAnalysis*piZeroAnalysis*electronAnalyzerSequence)
_egammaDQMOfflineHGCal = egammaDQMOffline.copy()
_egammaDQMOfflineHGCal += stdPhotonAnalysisHGCal

phase2_hgcal.toReplaceWith(
  egammaDQMOffline, _egammaDQMOfflineHGCal
)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
for e in [peripheralPbPb, pp_on_AA, pp_on_XeXe_2017]:
    e.toModify(stdPhotonAnalysis, phoProducer = cms.InputTag('islandPhotons'))
