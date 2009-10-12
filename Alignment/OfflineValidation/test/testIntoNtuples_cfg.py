import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_31X_V9::All"
#process.prefer("GlobalTag")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
									destinations = cms.untracked.vstring('detailedInfo', 
																		 'cout')
)

from CondCore.DBCommon.CondDBSetup_cfi import *
process.ZeroGeom = cms.ESSource("PoolDBESSource",CondDBSetup,
								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentRcd'),
												tag = cms.string('TrackerIdealGeometry210_mc')
								))
							)
process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentErrorRcd'),
												tag = cms.string('TrackerIdealGeometryErrors210_mc')
											))
								)
process.es_prefer_ZeroGeom = cms.ESPrefer("PoolDBESSource", "ZeroGeom")
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
									   input = cms.untracked.int32(0)
)
process.dump = cms.EDFilter("TrackerGeometryIntoNtuples",
							outputFile = cms.untracked.string('testInputGeometry.root'),
							outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)


