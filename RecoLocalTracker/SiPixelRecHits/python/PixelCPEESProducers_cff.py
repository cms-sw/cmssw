import FWCore.ParameterSet.Config as cms

#
# Load all Pixel Cluster Position Estimator ESProducers
#
#
# 1. RecHits using angles from module position
#
from RecoLocalTracker.SiPixelRecHits.PixelCPEInitial_cfi import *
#
# 2. TrackingRechits using angles from tracks
#
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#
# 3. Template algorithm
#
from RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi import *
#
# 4. Pixel Generic CPE
#
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
#
# 5. The new ESProducer for the Magnetic-field dependent template record
#
from CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi import *

