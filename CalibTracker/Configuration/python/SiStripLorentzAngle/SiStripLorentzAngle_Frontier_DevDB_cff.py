import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siStripLorentzAngle = copy.deepcopy(poolDBESSource)
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
siStripLorentzAngle.connect = 'frontier://FrontierDev/CMS_COND_STRIP'
siStripLorentzAngle.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripLorentzAngleRcd'),
    tag = cms.string('SiStripLorentzAngle_Ideal_20X')
))
siStripLorentzAngle.BlobStreamerName = 'TBufferBlobStreamingService'

