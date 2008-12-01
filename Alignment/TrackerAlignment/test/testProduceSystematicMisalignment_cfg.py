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
process.misalign = cms.EDAnalyzer("TrackerSystematicMisalignments",
								  # grab an existing geometry
								  fromDBGeom = cms.untracked.bool(True),
								  # flags	
								  radialFlag = cms.untracked.bool(True),
								  telescopeFlag = cms.untracked.bool(False),
								  layerRotFlag = cms.untracked.bool(False),
								  bowingFlag = cms.untracked.bool(False),
								  zExpFlag = cms.untracked.bool(False),
								  twistFlag = cms.untracked.bool(False),	
								  ellipticalFlag = cms.untracked.bool(False),
								  skewFlag = cms.untracked.bool(False),
								  saggitaFlag = cms.untracked.bool(False),
								  #epsilons
								  radialEpsilon = cms.untracked.double(5e-3), # default 5e-4 ~ 600 um
								  telescopeEpsilon = cms.untracked.double(1e-2), # default 5e-4 ~ 600 um
								  layerRotEpsilon = cms.untracked.double(9.43e-5), # 9.43e-6
								  bowingEpsilon = cms.untracked.double(9e-8), #6.77e-9
								  zExpEpsilon = cms.untracked.double(5e-3), # 2.02e-4
								  twistEpsilon = cms.untracked.double(2e-5),	# 2.04e-6
								  ellipticalEpsilon = cms.untracked.double(5e-3), # 5e-4
								  skewEpsilon = cms.untracked.double(5.5e-1), # 5.5e-2
								  saggitaEpsilon = cms.untracked.double(5.0e-3) #5.0e-4
								  )
	
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

process.p = cms.Path( process.misalign )


