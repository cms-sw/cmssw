import FWCore.ParameterSet.Config as cms

#
# Load all Pixel Cluster Position Estimator ESProducers
#
# 1. Template algorithm
#
from RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi import *
#
# 2. Pixel Generic CPE
#
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducer_cfi import *#import pixelCPEFastESProducer_cfi as PixelCPEFastESProducer
from RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducerPhase2_cfi import *#import pixelCPEFastESProducerPhase2_cfi as PixelCPEFastESProducerPhase2
#
# 3. ESProducer for the Magnetic-field dependent template records
#
from CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi import *
from CalibTracker.SiPixelESProducers.SiPixel2DTemplateDBObjectESProducer_cfi import *

