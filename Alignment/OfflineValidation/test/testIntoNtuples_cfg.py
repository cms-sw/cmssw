import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
									destinations = cms.untracked.vstring('detailedInfo', 
																		 'cout')
)

from CondCore.DBCommon.CondDBSetup_cfi import *
process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
							   connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
							   timetype = cms.string("runnumber"),
							   toGet = cms.VPSet(
												 cms.PSet(
														  record = cms.string('TrackerAlignmentRcd'),
														  tag = cms.string('Tracker10pbScenario210_mc')
														  ), 
												 cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
														  tag = cms.string('Tracker10pbScenarioErrors210_mc')
														  ))
							   )


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
									   input = cms.untracked.int32(0)
)
process.dump = cms.EDFilter("TrackerGeometryIntoNtuples",
							outputFile = cms.untracked.string('outputTree.root'),
							outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)


