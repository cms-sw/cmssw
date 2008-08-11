import FWCore.ParameterSet.Config as cms

# Dummy Lorentz angle producer (no data is put in the ES!)
from CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
# alternative
#include "RecoLocalTracker/SiStripRecHitConverter/data/StripCPE.cfi"
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

