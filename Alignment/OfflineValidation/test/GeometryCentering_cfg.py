import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.Geometry.GeometryDB_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']

### needed to get the geometry
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,"auto:run2_design")

from CondCore.CondDB.CondDB_cfi import *

CondDBAlignment = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
                            
### example starting MC geometry that is known to be off-centered
process.ZeroGeom = cms.ESSource("PoolDBESSource",
                                CondDBAlignment,
                                timetype = cms.string("runnumber"),
                                toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                           tag = cms.string('TrackerAlignment_Asymptotic_Run2016_v1_mc')
                                                           )
                                                  )                                                            
                                )
				
CondDBAPE = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))

process.ZeroAPE = cms.ESSource("PoolDBESSource",
                               CondDBAPE,
                               timetype = cms.string("runnumber"),
                               toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                          tag = cms.string('TrackerIdealGeometryErrorsExtended210_mc')
                                                          )
                                                 )
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
process.TrackerGeometryCompare.setCommonTrackerSystem = "TPBBarrel"  # for MC
#process.TrackerGeometryCompare.setCommonTrackerSystem = "TOBBarrel"  # for Data
process.TrackerGeometryCompare.levels = []
process.TrackerGeometryCompare.writeToDB = True

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("TPBCenteredOutputComparison.root")  # for MC
                                   #fileName=cms.string("TOBCenteredOutputComparison.root") #for Data
                                   )

CondDBoutput=CondDB.clone(connect = cms.string('sqlite_file:TrackerAlignment_Asymptotic_Run2016_v1_mc_Centred.db'))

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBoutput,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                     tag = cms.string('Alignments')
                                                                     ),
                                                            cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                                     tag = cms.string('AlignmentErrorsExtended')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.dump*process.TrackerGeometryCompare)

