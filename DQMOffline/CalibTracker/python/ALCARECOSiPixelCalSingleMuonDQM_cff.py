import FWCore.ParameterSet.Config as cms
from DQMOffline.CalibTracker.SiPixelCalSingleMuonAnalyzer_cfi import siPixelCalSingleMuonAnalyzer as alcaAnalyzer

#---------------
# AlCaReco DQM #
#---------------

__selectionName = 'SiPixelCalSingleMuonTight'
ALCARECOSiPixelCalSingleMuonTightSpecificDQM = alcaAnalyzer.clone(clusterCollection = 'ALCARECO'+__selectionName,
                                                                  nearByClusterCollection = 'closebyPixelClusters',
                                                                  trajectoryInput = 'ALCARECO'+__selectionName+'TracksRefit',  #making usage of what exists in Calibration/TkAlCaRecoProducers/python/ALCARECOSiPixelCalSingleMuonTight_cff.py
                                                                  muonTracks = 'ALCARECO'+__selectionName,
                                                                  dqmPath = "AlCaReco/"+__selectionName)
#------------
# Sequence #
#------------
ALCARECOSiPixelCalSingleMuonTightDQM = cms.Sequence(ALCARECOSiPixelCalSingleMuonTightSpecificDQM)
