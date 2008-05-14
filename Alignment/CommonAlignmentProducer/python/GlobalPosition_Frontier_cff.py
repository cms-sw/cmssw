# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_ALIGNMENT"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
GlobalPosition = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
GlobalPosition.connect = 'frontier://FrontierProd/CMS_COND_20X_ALIGNMENT'
# replace GlobalPosition.connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT" # oracle integration
GlobalPosition.toGet = cms.VPSet(cms.PSet(
    record = cms.string('GlobalPositionRcd'),
    tag = cms.string('IdealGeometry')
))

