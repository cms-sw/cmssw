import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# TransientTrackingBuilder
ttrhb4GlobalPixelSeeds = copy.deepcopy(ttrhbwr)
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.PixelTrackSeeds_cfi import *
ttrhb4GlobalPixelSeeds.StripCPE = 'Fake'
ttrhb4GlobalPixelSeeds.ComponentName = 'TTRHBuilder4GlobalPixelSeeds'

