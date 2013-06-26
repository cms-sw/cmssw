import FWCore.ParameterSet.Config as cms

# Dummy Lorentz angle producer (no data is put in the ES!)
from CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi import *    
import CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi
siStripLAFakeESSourceforSimulation = CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi.siStripLorentzAngleFakeESSource.clone()
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
# alternative
#include "RecoLocalTracker/SiStripRecHitConverter/data/StripCPE.cfi"
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
es_prefer_siStripLAFakeESSource = cms.ESPrefer("SiStripLAFakeESSource","siStripLAFakeESSource")
siStripLAFakeESSourceforSimulation.appendToDataLabel = 'fake'


