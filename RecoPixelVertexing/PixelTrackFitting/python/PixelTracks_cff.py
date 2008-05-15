import FWCore.ParameterSet.Config as cms

# Magntic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# Geometry (all CMS)
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# Tracker Geometry Builder
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# Tracker Numbering Builder
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# Reco geometry 
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# for Transient rechits?
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#-ap   include "CalibTracker/Configuration/data/SiPixelLorentzAngle/SiPixelLorentzAngle_Fake.cff"
# include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngle = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone()
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
myTTRHBuilderWithoutAngle.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle.ComponentName = 'PixelTTRHBuilderWithoutAngle'

