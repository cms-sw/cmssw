import FWCore.ParameterSet.Config as cms

# from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
#from CalibTracker.SiStripLorentzAngle.SiStripLAFakeESSource_cfi import *
#from CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff import *
#from Configuration.StandardSequences.L1Emulator_cff import *
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi import *
from RecoMuon.TrackerSeedGenerator.TSGFromL1_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngleSeedsFromL1Muon = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngleSeedsFromL1Muon'
)
