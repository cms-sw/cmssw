import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.Tree_ALCARECO_cff import *
LorentzAngleTracks.src = 'generalTracks'

#local reconstruction
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
recolocal = cms.Sequence( siPixelDigis*siPixelClusters*
                          siStripDigis*siStripZeroSuppression*siStripClusters)
siPixelDigis.InputLabel = 'rawDataCollector'

#tracking
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
recotrack = cms.Sequence( offlineBeamSpot + siPixelRecHits*siStripMatchedRecHits*recopixelvertexing*ckftracks)

#Schedule
reconstruction_step = cms.Path( recolocal + recotrack )
schedule = cms.Schedule( reconstruction_step, filter_refit_ntuplize_step )
