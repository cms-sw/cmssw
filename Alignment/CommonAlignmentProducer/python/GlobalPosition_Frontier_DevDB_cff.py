# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_ALIGNMENT"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
GlobalPosition = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
GlobalPosition.connect = 'frontier://FrontierDev/CMS_COND_ALIGNMENT'
GlobalPosition.toGet = cms.VPSet(cms.PSet(
    record = cms.string('GlobalPositionRcd'),
    tag = cms.string('IdealGeometry')
))

