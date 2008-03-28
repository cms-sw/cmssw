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
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelHitPairTracks_cfi import *
ttrhbwr.StripCPE = 'Fake'

