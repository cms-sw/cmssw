# The following comments couldn't be translated into the new config version:

#cms_conditions_data/CMS_COND_20X_ALIGNMENT"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
GlobalPosition = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
GlobalPosition.connect = 'frontier://cms_conditions_data/CMS_COND_20X_ALIGNMENT'
GlobalPosition.toGet = cms.VPSet(cms.PSet(
    record = cms.string('GlobalPositionRcd'),
    tag = cms.string('IdealGeometry')
))

