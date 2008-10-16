import FWCore.ParameterSet.Config as cms

# Reading from DB
from CondCore.DBCommon.CondDBCommon_cfi import *
PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TibTidTecAllSurvey_v2')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('TibTidTecAllSurveyAPE_v2')
        ))
)

#
# Survey info corrections for the TIF tracker setup
# It assumes the standard geometry builder is used:
# include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
TrackerDigiGeometryESModule.applyAlignment = True
CondDBCommon.connect = 'oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT' ##cms_orcoff_int2r/CMS_COND_ALIGNMENT"

CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

