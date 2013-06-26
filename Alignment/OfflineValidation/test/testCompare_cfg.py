import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "FT_R_53_V6C::All" 
process.GlobalTag.globaltag = "GR_R_61_V6::All" # FT_R_42_V13A used in rereco of 2010 cosmics

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")


process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.load("DQM.SiStripCommon.TkHistoMap_cfi")
#process.TkDetMap = cms.Service("TkDetMap")
#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("DQMServices.Core.DQMStore_cfg") 
#process.DQMStore=cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.TrackerGeometryCompare)

process.TrackerGeometryCompare.inputROOTFile1 = 'myInputGeometry_mp1260m2.root' 
process.TrackerGeometryCompare.inputROOTFile2 = 'IDEAL' 

process.TrackerGeometryCompare.outputFile = 'outputComparison_mp1260m2_2012AB.root'

process.load("CommonTools.UtilAlgos.TFileService_cfi")  
#process.TFileService = cms.Service("TFileService",
#		fileName = cms.string('TkSurfDeform_mp1260m2_2012AB.root') 
#		)
process.TFileService.fileName = cms.string('TkSurfDeform_mp1260m2_2012AB.root') 

process.TrackerGeometryCompare.levels = ['DetUnit']  
#process.TrackerGeometryCompare.levels = ['Tracker','DetUnit'] 
#process.TrackerGeometryCompare.levels = ['Tracker','TOBBarrel','TIBBarrel','TECEndcap','TECDisk']
#process.TrackerGeometryCompare.levels = ["TPBBarrel","TPEEndcap","TIBBarrel","TIDEndcap","TOBBarrel","TECEndcap"] 

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
