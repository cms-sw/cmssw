import FWCore.ParameterSet.Config as cms

#
# Fake for condition data that are not in DB
#
from CalibTracker.SiStripESProducers.SiStripPedestalsFakeSource_cfi import *
from CalibTracker.SiStripESProducers.SiStripQualityFakeESSource_cfi import *
#
# Dependent Records
#
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#Gain
from CalibTracker.SiStripESProducers.SiStripGainFakeSource_cfi import *
#SiStripGainFakeESSource.appendToDataLabel = 'fake'
import CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi
siStripGainESProducerforSimulation = CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi.siStripGainESProducer.clone()
siStripGainESProducerforSimulation.appendToDataLabel = 'fake'
siStripGainESProducerforSimulation.APVGain = 'fakeAPVGain'

#Lorentz Angle
import CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi
siStripLAFakeESSourceforSimulation = CalibTracker.SiStripESProducers.SiStripLAFakeESSource_cfi.siStripLAFakeESSource.clone()
siStripLAFakeESSourceforSimulation.appendToDataLabel = 'fake'

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

#cabling
sistripconn = cms.ESProducer("SiStripConnectivity")

TrackerDigiGeometryESModule.applyAlignment = True



