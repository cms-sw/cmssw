import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
GlobalPosition = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_COND_31X_ALIGNMENT',
    #connect = 'frontier://FrontierPrep/CMS_COND_ALIGNMENT',
    #connect = 'frontier://FrontierPrep/CMS_COND_PRESH', #temp. hack for 310pre4!
    toGet = cms.VPSet(cms.PSet(record = cms.string('GlobalPositionRcd'),
                               tag = cms.string('IdealGeometry')
                               )
                      )
    )

