import FWCore.ParameterSet.Config as cms

# local reconstruction
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
recolocal = cms.Sequence(siPixelDigis + siPixelClustersPreSplitting + siStripDigis + siStripZeroSuppression + siStripClusters)
siPixelDigis.cpu.InputLabel = 'rawDataCollector'

# tracking
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.Configuration.RecoPixelVertexing_cff import *
recotrack = cms.Sequence(offlineBeamSpot + siPixelRecHitsPreSplitting + siStripMatchedRecHits + recopixelvertexing + ckftracks)
recotrackP5 = cms.Sequence(offlineBeamSpot + siPixelRecHitsPreSplitting + siStripMatchedRecHits + recopixelvertexing + ctftracksP5)

# Schedule
reconstruction_step = cms.Path(recolocal + recotrack)
reconstructionP5_step = cms.Path(recolocal + recotrackP5)
