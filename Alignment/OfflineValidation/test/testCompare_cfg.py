import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT0831X_V4::All"

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")


process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.TrackerGeometryCompare)
process.TrackerGeometryCompare.inputROOTFile1 = 'CRAFT08_vs_IDEAL.Comparison_commonLasSubDets.root'
process.TrackerGeometryCompare.inputROOTFile2 = 'MillePede_LAS_CRAFT08_vs_IDEAL.Comparison_commonLasSubDets.root'
process.TrackerGeometryCompare.outputFile = 'outputComparison.root'
process.TrackerGeometryCompare.levels = ['Tracker','TOBBarrel','TIBBarrel','TECEndcap','TECDisk']
#process.TrackerGeometryCompare.levels = ['DetUnit']

"""
process.TrackerGeometryCompare.writeToDB = True

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
   process.CondDBSetup,
   timetype = cms.untracked.string('runnumber'),
   connect = cms.string('sqlite_file:yourNewTrackerCenteredObject.db'),
   toPut = cms.VPSet(cms.PSet(
       record = cms.string('TrackerAlignmentRcd'),
       tag = cms.string('Alignments')
   ),
       cms.PSet(
           record = cms.string('TrackerAlignmentErrorRcd'),
           tag = cms.string('AlignmentErrors')
       ))
)
"""
