# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_STRIP"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripLorentzAngle = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
#replace siStripLorentzAngle.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
#replace siStripLorentzAngle.DBParameters.messageLevel=2
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
# alternative
#include "RecoLocalTracker/SiStripRecHitConverter/data/StripCPE.cfi"
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
siStripLorentzAngle.connect = 'frontier://FrontierProd/CMS_COND_20X_STRIP'
siStripLorentzAngle.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripLorentzAngleRcd'),
    tag = cms.string('SiStripLorentzAngle_Ideal_20X')
))
siStripLorentzAngle.BlobStreamerName = 'TBufferBlobStreamingService'

