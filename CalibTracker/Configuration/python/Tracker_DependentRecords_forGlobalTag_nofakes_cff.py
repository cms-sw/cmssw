import FWCore.ParameterSet.Config as cms

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
import CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi
siStripLAFakeESSourceforSimulation = CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi.siStripLorentzAngleFakeESSource.clone()

siStripLAFakeESSourceforSimulation.appendToDataLabel = 'fake'

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

#cabling
sistripconn = cms.ESProducer("SiStripConnectivity")

TrackerDigiGeometryESModule.applyAlignment = True


from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import *
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") )
     )


