# The following comments couldn't be translated into the new config version:

#orcon/CMS_COND_STRIP"
#orcon/CMS_COND_STRIP"
#orcon/CMS_COND_STRIP"
#orcon/CMS_COND_STRIP"
import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
# Tracker alignment
#
#include "CalibTracker/Configuration/data/TrackerAlignment/TrackerAlignment_Frontier_IntDB.cff"
#
# Strip 
#
#
#Cabling
#
siStripFedCabling = copy.deepcopy(poolDBESSource)
import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
#
#
#Gain
#
siStripApvGain = copy.deepcopy(poolDBESSource)
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
#
#LA
#
from CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff import *
import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
#
#Noise
#
siStripNoise = copy.deepcopy(poolDBESSource)
import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
#
#Pedestals
#
siStripPedestals = copy.deepcopy(poolDBESSource)
sistripconn = cms.ESProducer("SiStripConnectivity")

#
# SiStripQuality
#
a = cms.ESSource("PoolDBESSource",
    appendToDataLabel = cms.string('SiStripBadModule_v1'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('SiStripBadModule_VenturiList_TIF_20X')
    )),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://orcon/CMS_COND_STRIP'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

b = cms.ESSource("PoolDBESSource",
    appendToDataLabel = cms.string('SiStripBadFiber_v1'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('SiStripBadFiber_VenturiList_TIF_20X')
    )),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://orcon/CMS_COND_STRIP'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    appendToDataLabel = cms.string('test1'),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('SiStripBadModule_v1')
    ), 
        cms.PSet(
            record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('SiStripBadFiber_v1')
        ))
)

siStripFedCabling.connect = 'oracle://orcon/CMS_COND_STRIP'
siStripFedCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('SiStripFedCabling_TIF_20X')
))
siStripFedCabling.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
siStripApvGain.connect = 'oracle://orcon/CMS_COND_STRIP'
siStripApvGain.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripApvGainRcd'),
    tag = cms.string('SiStripApvGain_TickMarksfromASCII_TIF_20X')
))
siStripApvGain.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
siStripNoise.connect = 'oracle://orcon/CMS_COND_STRIP'
siStripNoise.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripNoisesRcd'),
    tag = cms.string('SiStripNoise_TIF_20X')
))
siStripNoise.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
siStripPedestals.connect = 'oracle://orcon/CMS_COND_STRIP'
siStripPedestals.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripPedestalsRcd'),
    tag = cms.string('SiStripPedestals_TIF_20X')
))
siStripPedestals.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

