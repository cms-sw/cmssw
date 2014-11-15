import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "DESIGN311_V2::All"#"DESIGN_3X_V25::All"

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
#								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
#								timetype = cms.string("runnumber"),
#								toGet = cms.VPSet(
#											cms.PSet(
#												record = cms.string('TrackerAlignmentRcd'),
#												tag = cms.string('TrackerIdealGeometry210_mc')
#								))                                                                                                
                                                                connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp0610/jobData/fixedhalfdiskangles/alignments_MP.db'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentRcd'),
												tag = cms.string('Alignments')
								))                                                                                                
							)
							
process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentErrorExtendedRcd'),
												tag = cms.string('TrackerIdealGeometryErrors210_mc')
											))
                               )
process.es_prefer_ZeroGeom = cms.ESPrefer("PoolDBESSource", "ZeroGeom")
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

GeometryIntoNtuplesRootFile = cms.untracked.string('InputGeometry.root')

# into ntuples
process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
			       outputFile = GeometryIntoNtuplesRootFile,
			       outputTreename = cms.untracked.string('alignTree')
)


# geometry comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
process.TrackerGeometryCompare.inputROOTFile1 = 'IDEAL'
process.TrackerGeometryCompare.inputROOTFile2 = GeometryIntoNtuplesRootFile
process.TrackerGeometryCompare.outputFile = 'TOBCenteredOutputComparison.root'
process.TrackerGeometryCompare.setCommonTrackerSystem = "TOBBarrel"
process.TrackerGeometryCompare.levels = []
process.TrackerGeometryCompare.writeToDB = True


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
   process.CondDBSetup,
   timetype = cms.untracked.string('runnumber'),
   connect = cms.string('sqlite_file:CRAFT11fromEoY_TopCentred.db'),
   toPut = cms.VPSet(cms.PSet(
       record = cms.string('TrackerAlignmentRcd'),
       tag = cms.string('Alignments')
   ),
       cms.PSet(
           record = cms.string('TrackerAlignmentErrorExtendedRcd'),
           tag = cms.string('AlignmentErrorsExtended')
       ))
)

process.p = cms.Path(process.dump*process.TrackerGeometryCompare)

