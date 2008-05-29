import FWCore.ParameterSet.Config as cms

#
# Tracker Local Reco
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_SimData_cfi import * ##SimData.cfi compatible with RealData reco in 20x

from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_SimData2_cfi import * ##SimData2.cfi compatible with RealData reco in 20x

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
pixeltrackerlocalreco = cms.Sequence(siPixelClusters*siPixelRecHits)
striptrackerlocalreco = cms.Sequence(siStripZeroSuppression*siStripClusters*siStripMatchedRecHits)
trackerlocalreco = cms.Sequence(pixeltrackerlocalreco*striptrackerlocalreco)

