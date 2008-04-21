import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siStripApvGain = copy.deepcopy(poolDBESSource)
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
siStripApvGain.connect = 'frontier://FrontierDev/CMS_COND_STRIP'
siStripApvGain.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripApvGainRcd'),
    tag = cms.string('SiStripGain_Ideal_20X')
))
siStripApvGain.BlobStreamerName = 'TBufferBlobStreamingService'

