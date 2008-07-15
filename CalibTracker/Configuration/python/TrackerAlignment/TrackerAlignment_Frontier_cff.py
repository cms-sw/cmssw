# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff import *
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
trackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
trackerAlignment.connect = 'frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'
#trackerAlignment.connect = 'oracle://cms_orcoff_prep/CMS_COND_ALIGNMENT' # oracle dev. DB
trackerAlignment.toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('TrackerIdealGeometry210_mc')
), 
    cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
    ))
#from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")
# to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True

