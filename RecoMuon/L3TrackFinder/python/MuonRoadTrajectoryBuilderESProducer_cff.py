import FWCore.ParameterSet.Config as cms

#measurement tracker
# from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# with no pixel (optional)
#es_module MeasurementTrackerNoPixel = MeasurementTrackerESProducer  from "RecoTracker/MeasurementDet/data/MeasurementTrackerESProducer.cfi"
#replace MeasurementTrackerNoPixel.ComponentName = "MeasurementTrackerNoPixel"
#replace MeasurementTrackerNoPixel.pixelClusterProducer = ""
#stepping helix propagator anydirection
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from RecoMuon.L3TrackFinder.MuonRoadTrajectoryBuilderESProducer_cfi import *


