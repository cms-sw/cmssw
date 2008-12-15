import FWCore.ParameterSet.Config as cms

#
# Fake for condition data that are not in DB
#
from CalibTracker.SiStripESProducers.fake.SiStripPedestalsFakeESSource_cfi import *
#from CalibTracker.SiStripESProducers.SiStripQualityFakeESSource_cfi import *
#
# Dependent Records
#
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#Gain
# first SiStripGainESProducer takes SiStripGainRcd from DB
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
#second SiStripGainESProducer is used in the digitizer and takes SiStripGainRcd from SiStripGainFakeSource
from CalibTracker.SiStripESProducers.fake.SiStripApvGainFakeESSource_cfi import *
siStripApvGainFakeESSource.appendToDataLabel = 'fakeAPVGain'
import CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi
siStripGainESProducerforSimulation = CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi.siStripGainESProducer.clone()
siStripGainESProducerforSimulation.appendToDataLabel = 'fake'
siStripGainESProducerforSimulation.APVGain = 'fakeAPVGain'

#Lorentz Angle
#this SiStripLAFakeESSource is used by the digitizer
from CalibTracker.SiStripESProducers.services.SiStripLorentzAngleGeneratorService_cfi import *
import CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi
siStripLAFakeESSourceforSimulation = CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi.siStripLorentzAngleFakeESSource.clone()

siStripLAFakeESSourceforSimulation.appendToDataLabel = 'fake'

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

#cabling
sistripconn = cms.ESProducer("SiStripConnectivity")

TrackerDigiGeometryESModule.applyAlignment = True

##add quality info
#NB in case of usage of GlobalTag, these Fakes will be substituded by the records in the GT (having the esprefer)
from CalibTracker.SiStripESProducers.fake.SiStripBadModuleFakeESSource_cfi import *
from CalibTracker.SiStripESProducers.fake.SiStripBadFiberFakeESSource_cfi import *
from CalibTracker.SiStripESProducers.fake.SiStripBadChannelFakeESSource_cfi import *

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import *
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") ),
     )


