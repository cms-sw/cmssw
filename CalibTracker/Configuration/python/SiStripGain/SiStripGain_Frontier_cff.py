# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_STRIP"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripApvGain = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
#replace siStripApvGain.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
#replace siStripApvGain.DBParameters.messageLevel=2
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
siStripApvGain.connect = 'frontier://FrontierProd/CMS_COND_20X_STRIP'
siStripApvGain.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripApvGainRcd'),
    tag = cms.string('SiStripGain_Ideal_20X')
))
siStripApvGain.BlobStreamerName = 'TBufferBlobStreamingService'

