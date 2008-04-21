import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_DevDB_cff import *
import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
trackerAlignment = copy.deepcopy(poolDBESSource)
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")
trackerAlignment.connect = 'frontier://FrontierDev/CMS_COND_ALIGNMENT'
trackerAlignment.toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('TrackerIdealGeometry200')
), 
    cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors200')
    ))
TrackerDigiGeometryESModule.applyAlignment = True

