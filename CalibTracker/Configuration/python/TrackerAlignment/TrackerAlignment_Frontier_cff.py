# This is a template in case constants are not taken from global tag:
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff import *
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# 'connect' to be adjusted to release, 'tags' to newer payloads... 
trackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    # connect = 'sqlite_file:afile.db'
    connect = 'frontier://FrontierProd/CMS_COND_21X_ALIGNMENT',
    toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                               tag = cms.string('TrackerIdealGeometry210_mc')
                               ), 
                      cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                               tag = cms.string('TrackerIdealGeometryErrors210_mc')
                               ))
    )

# Might be needed if you want to overwrite the global tag:
# ESPrefer("PoolDBESSource", "trackerAlignment")
