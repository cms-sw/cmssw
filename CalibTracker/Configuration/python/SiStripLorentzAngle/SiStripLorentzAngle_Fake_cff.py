import FWCore.ParameterSet.Config as cms

# Dummy Lorentz angle producer (no data is put in the ES!)
from CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi import *
import CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi
siStripLAFakeESSourceforSimulation = CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi.siStripLAFakeESSource.clone()
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
# alternative
#include "RecoLocalTracker/SiStripRecHitConverter/data/StripCPE.cfi"
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
es_prefer_siStripLAFakeESSource = cms.ESPrefer("SiStripLAFakeESSource","siStripLAFakeESSource")
siStripLAFakeESSourceforSimulation.appendToDataLabel = 'fake'


