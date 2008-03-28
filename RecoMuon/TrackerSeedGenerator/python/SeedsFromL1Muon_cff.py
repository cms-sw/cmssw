import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from CalibTracker.SiStripLorentzAngle.SiStripLAFakeESSource_cfi import *
from Configuration.StandardSequences.L1Emulator_cff import *
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
from RecoMuon.TrackerSeedGenerator.SeedsFromL1Muon_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
myTTRHBuilderWithoutAngleSeedsFromL1Muon = copy.deepcopy(ttrhbwr)
myTTRHBuilderWithoutAngleSeedsFromL1Muon.StripCPE = 'Fake'
myTTRHBuilderWithoutAngleSeedsFromL1Muon.ComponentName = 'PixelTTRHBuilderWithoutAngleSeedsFromL1Muon'

