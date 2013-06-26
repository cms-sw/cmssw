import FWCore.ParameterSet.Config as cms

# includes for tracking rechits
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *

# modules to produce standard tracker seeds
from RecoTracker.Configuration.RecoTracker_cff import *
# module to produce electron seeds
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cfi import *

