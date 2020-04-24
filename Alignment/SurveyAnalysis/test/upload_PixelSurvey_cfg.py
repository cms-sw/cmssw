import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.tracker = cms.EDFilter("SurveyInputTrackerFromDB",
	textFileName = cms.string('../data/FPixBPix_Survey.txt')
)

from CondCore.DBCommon.CondDBSetup_cfi import *
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:FPixBPix_Survey.db'),
    toPut = cms.VPSet(cms.PSet(
				record = cms.string('TrackerAlignmentRcd'),
				tag = cms.string('Alignments')
			), 
			cms.PSet(
				record = cms.string('TrackerAlignmentErrorExtendedRcd'),
				tag = cms.string('AlignmentErrorsExtended')
			))
)

process.p = cms.Path(process.tracker)


