# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_PIXEL"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siPixelLorentzAngle = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelLorentzAngle.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelLorentzAngle.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelLorentzAngleRcd'),
    tag = cms.string('trivial_LorentzAngle_mc')
))

