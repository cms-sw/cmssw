import FWCore.ParameterSet.Config as cms

#Gain
# first SiStripGainESProducer takes SiStripGainRcd from DB
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *

#LorentzAngle
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *

#cabling
sistripconn = cms.ESProducer("SiStripConnectivity")

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import *
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadModuleRcd"),   tag    = cms.string("") )
     )


