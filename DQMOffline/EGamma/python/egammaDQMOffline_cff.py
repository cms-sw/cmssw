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


stdPhotonAnalysis = DQMOffline.EGamma.photonAnalyzer_cfi.photonAnalysis.clone()
stdPhotonAnalysis.ComponentName = cms.string('stdPhotonAnalysis')
stdPhotonAnalysis.analyzerName = cms.string('stdPhotonAnalyzer')
stdPhotonAnalysis.phoProducer = cms.InputTag('photons')
stdPhotonAnalysis.OutputMEsInRootFile = cms.bool(False)
stdPhotonAnalysis.Verbosity = cms.untracked.int32(0)
stdPhotonAnalysis.standAlone = cms.bool(False)

piZeroAnalysis.OutputMEsInRootFile = cms.bool(False)
piZeroAnalysis.Verbosity = cms.untracked.int32(0)
piZeroAnalysis.standAlone = cms.bool(False)


zmumugammaOldAnalysis = DQMOffline.EGamma.zmumugammaAnalyzer_cfi.zmumugammaAnalysis.clone()
zmumugammaOldAnalysis.ComponentName = cms.string('zmumugammaOldAnalysis')
zmumugammaOldAnalysis.analyzerName = cms.string('zmumugammaOldValidation')
zmumugammaOldAnalysis.phoProducer = cms.InputTag('photons')

# HGCal customizations
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
stdPhotonAnalysisHGCal = stdPhotonAnalysis.clone()
stdPhotonAnalysisHGCal.ComponentName = 'stdPhotonAnalyzerHGCalFromMultiCl'
stdPhotonAnalysisHGCal.analyzerName = 'stdPhotonAnalyzerHGCalFromMultiCl'
stdPhotonAnalysisHGCal.phoProducer = 'photonsFromMultiCl'
stdPhotonAnalysisHGCal.isolationStrength = 2
stdPhotonAnalysisHGCal.etaMin = -3.0
stdPhotonAnalysisHGCal.etaMax = 3.0
stdPhotonAnalysisHGCal.maxPhoEta = 3.0

egammaDQMOffline = cms.Sequence(photonAnalysis*stdPhotonAnalysis*zmumugammaOldAnalysis*zmumugammaAnalysis*piZeroAnalysis*electronAnalyzerSequence)
_egammaDQMOfflineHGCal = egammaDQMOffline.copy()
_egammaDQMOfflineHGCal += stdPhotonAnalysisHGCal

phase2_hgcal.toReplaceWith(
  egammaDQMOffline, _egammaDQMOfflineHGCal
)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
for e in [peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017]:
    e.toModify(stdPhotonAnalysis, phoProducer = cms.InputTag('islandPhotons'))
