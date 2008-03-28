import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
GlobalPosition = copy.deepcopy(poolDBESSource)
GlobalPosition.connect = 'frontier://FrontierDev/CMS_COND_ALIGNMENT'
GlobalPosition.toGet = cms.VPSet(cms.PSet(
    record = cms.string('GlobalPositionRcd'),
    tag = cms.string('IdealGeometry')
))

