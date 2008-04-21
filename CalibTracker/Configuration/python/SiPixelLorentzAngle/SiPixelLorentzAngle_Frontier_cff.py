import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siPixelLorentzAngle = copy.deepcopy(poolDBESSource)
siPixelLorentzAngle.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelLorentzAngle.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelLorentzAngleRcd'),
    tag = cms.string('trivial_LorentzAngle_mc')
))

