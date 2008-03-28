# The following comments couldn't be translated into the new config version:

# FRONTIER

import FWCore.ParameterSet.Config as cms

#
# Take muon alignment corrections from Frontier
#
from CondCore.DBCommon.CondDBSetup_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
muonAlignment = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DTIdealGeometry200_mc')
    ), cms.PSet(
        record = cms.string('DTAlignmentErrorRcd'),
        tag = cms.string('DTIdealGeometryErrors200_mc')
    ), cms.PSet(
        record = cms.string('CSCAlignmentRcd'),
        tag = cms.string('CSCIdealGeometry200_mc')
    ), cms.PSet(
        record = cms.string('CSCAlignmentErrorRcd'),
        tag = cms.string('CSCIdealGeometryErrors200_mc')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_ALIGNMENT')
)

es_prefer_muonAlignment = cms.ESPrefer("PoolDBESSource","muonAlignment")
# to apply misalignments
DTGeometryESModule.applyAlignment = True
CSCGeometryESModule.applyAlignment = True

