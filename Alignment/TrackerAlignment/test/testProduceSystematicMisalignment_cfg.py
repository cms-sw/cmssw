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

# initial geom
# configure the database file - use survey one for default
from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerGeom = cms.ESSource("PoolDBESSource",
				CondDBSetup,
				timetype = cms.string('runnumber'),
				toGet = cms.VPSet(
								cms.PSet(
										record = cms.string('TrackerAlignmentRcd'),
										tag = cms.string('Alignments')
								), 
								cms.PSet(
										record = cms.string('TrackerAlignmentErrorRcd'),
										tag = cms.string('AlignmentErrors')
								)),
				connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/alignmentRcds/AllSurveyButTIBTIDGlobal_FPixBPix_wErrs_v2.db')
)
										
											
# input
process.load("Alignment.TrackerAlignment.TrackerSystematicMisalignments_cfi")
process.TrackerSystematicMisalignments.fromDBGeom = True
process.TrackerSystematicMisalignments.radialEpsilon = 5e-4 # default 5e-4 ~ 600 um
	
# output
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
										  CondDBSetup,
										  toPut = cms.VPSet(cms.PSet(
																	 record = cms.string('TrackerAlignmentRcd'),
																	 tag = cms.string('Alignments')
																	 ), 
															cms.PSet(
																	 record = cms.string('TrackerAlignmentErrorRcd'),
																	 tag = cms.string('AlignmentErrors')
																	 )),
										  connect = cms.string('sqlite_file:test.db')
)

process.p = cms.Path( process.TrackerSystematicMisalignments )


