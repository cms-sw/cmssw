import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonOfflineClient_cfi import *
import DQMOffline.EGamma.photonOfflineClient_cfi


stdPhotonOfflineClient = DQMOffline.EGamma.photonOfflineClient_cfi.photonOfflineClient.clone()
stdPhotonOfflineClient.ComponentName = cms.string('stdPhotonOfflineClient')
stdPhotonOfflineClient.analyzerName = cms.string('stdPhotonAnalyzer')

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
stdPhotonOfflineClientHGCal = stdPhotonOfflineClient.clone()
stdPhotonOfflineClientHGCal.analyzerName = 'stdPhotonAnalyzerHGCalFromMultiCl'
from DQMOffline.EGamma.egammaDQMOffline_cff import stdPhotonAnalysisHGCal
stdPhotonOfflineClientHGCal.etaBin = stdPhotonAnalysisHGCal.etaBin
stdPhotonOfflineClientHGCal.etaMin = stdPhotonAnalysisHGCal.etaMin
stdPhotonOfflineClientHGCal.etaMax = stdPhotonAnalysisHGCal.etaMax

photonOfflineDQMClient = cms.Sequence(photonOfflineClient*stdPhotonOfflineClient)
_photonOfflineDQMClientHGCal = photonOfflineDQMClient.copy()
_photonOfflineDQMClientHGCal += stdPhotonOfflineClientHGCal

phase2_hgcal.toReplaceWith(
  photonOfflineDQMClient, _photonOfflineDQMClientHGCal
)

