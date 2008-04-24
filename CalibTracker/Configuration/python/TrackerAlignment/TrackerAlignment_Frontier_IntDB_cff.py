# The following comments couldn't be translated into the new config version:

#cms_conditions_data/CMS_COND_20X_ALIGNMENT"
# FIXME: Should we use that?
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_IntDB_cff import *
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
trackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")
trackerAlignment.connect = 'frontier://cms_conditions_data/CMS_COND_20X_ALIGNMENT'
trackerAlignment.toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('TrackerIdealGeometry200')
), 
    cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors200')
    ))
# to apply misalignments
TrackerDigiGeometryESModule.applyAlignment = True

