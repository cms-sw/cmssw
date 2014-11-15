import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("CondCore.DBCommon.CondDBSetup_cfi")
from CondCore.DBCommon.CondDBSetup_cfi import *

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.tracker = cms.EDFilter("CreateSurveyRcds",
    inputGeom = cms.untracked.string('sqlite'),
	generatedRandom = cms.untracked.bool(False),
	generatedSimple = cms.untracked.bool(False),	
	simpleMis = cms.untracked.double(0.005)
)

process.uploader = cms.EDFilter("SurveyDBUploader",
    errorRcd = cms.string('TrackerSurveyErrorExtendedRcd'),
    valueRcd = cms.string('TrackerSurveyRcd')
)

# input
# need a POOLORA input geometry if the input geom is "sqlite"
# otherwise not needed
process.trackerAlignment = cms.ESSource("PoolDBESSource",
					 CondDBSetup,
					 timetype = cms.string('runnumber'),
					 toGet = cms.VPSet(cms.PSet(
												record = cms.string('TrackerAlignmentRcd'),
												tag = cms.string('Alignments')
												), 
									   cms.PSet(
												record = cms.string('TrackerAlignmentErrorExtendedRcd'),
												tag = cms.string('AlignmentErrorsExtended')
												)),
					 connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/alignmentRcds/AllSurveyButTIBTIDGlobal_FPixBPix_wErrs_v2.db')
)

#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

# output
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerSurveyRcd'),
        tag = cms.string('valueTag')
    ), 
        cms.PSet(
            record = cms.string('TrackerSurveyErrorExtendedRcd'),
            tag = cms.string('errorTag')
        )),
    connect = cms.string('sqlite_file:test.db')
)

process.p = cms.Path(process.tracker*process.uploader)


