
import FWCore.ParameterSet.Config as cms

#
# Take muon alignment corrections from Frontier
#
from CondCore.DBCommon.CondDBSetup_cfi import *
muonAlignment = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DTIdealGeometry200_mc')
    ), 
        cms.PSet(
            record = cms.string('DTAlignmentErrorRcd'),
            tag = cms.string('DTIdealGeometryErrors200_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentRcd'),
            tag = cms.string('CSCIdealGeometry200_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentErrorRcd'),
            tag = cms.string('CSCIdealGeometryErrors200_mc')
        )),
# FRONTIER
    connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT')
# ORACLE
   #connect = cms.string("oracle://cms_orcoff_prod/CMS_COND_21X_ALIGNMENT")

)
