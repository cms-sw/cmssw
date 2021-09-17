import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveReader")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V3::All'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)



###### OUTPUT HISTOGRAM FILE NAME #######
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )




##### DATABASE CONNECTION INFO ######
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel = 1

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(10),
    firstRun = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(0)
)



###### TAGS TO READ ######
process.PoolDBESSourceForReader = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
            record = cms.string('SiPixelFedCablingMapRcd'),
            tag = cms.string('SiPixelFedCablingMap_v14')
        ), 
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('trivial_LorentzAngle')
        ),
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleSimRcd'),
            tag = cms.string('SiPixelLorentzAngleSim_v02')
        ),
        cms.PSet(
            record = cms.string('SiPixelTemplateDBObjectRcd'),
            tag = cms.string('SiPixelTemplateDBObject')
        ),
        cms.PSet(
           record = cms.string('SiPixelQualityFromDbRcd'),
           tag = cms.string('SiPixelQuality_test')
        ),
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_const')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
        ),
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_const_new')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_hlt_const')
       ))
)






###### PREFER ABOVE TAGS #######
process.esprefer_DBReaders = cms.ESPrefer("PoolDBESSource", "PoolDBESSourceForReader")





####### GAIN READERS ######
process.SiPixelCondObjOfflineReader = cms.EDAnalyzer("SiPixelCondObjOfflineReader",
                                                   process.SiPixelGainCalibrationServiceParameters,
                                                   useSimRcd = cms.bool(False)
)

process.SiPixelCondObjOfflineSimReader = cms.EDAnalyzer("SiPixelCondObjOfflineReader",
                                                      process.SiPixelGainCalibrationServiceParameters,
                                                      useSimRcd = cms.bool(True)
)

process.SiPixelCondObjForHLTReader = cms.EDAnalyzer("SiPixelCondObjForHLTReader",
                                                  process.SiPixelGainCalibrationServiceParameters,
                                                  useSimRcd = cms.bool(False)
)

process.SiPixelCondObjForHLTSimReader = cms.EDAnalyzer("SiPixelCondObjForHLTReader",
                                                     process.SiPixelGainCalibrationServiceParameters,
                                                     useSimRcd = cms.bool(True)
)



####### LORENTZ ANGLE READERS ######
process.SiPixelLorentzAngleReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(False)
)

process.SiPixelLorentzAngleSimReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(True)
)



####### CABLING MAP READER ######
process.SiPixelFedCablingMapAnalyzer = cms.EDAnalyzer("SiPixelFedCablingMapAnalyzer")


#######  QUALITY READER #######
process.SiPixelBadModuleReader = cms.EDAnalyzer("SiPixelBadModuleReader")


####### TEMPLATE OBJECT READER ######
#Change to True if you would like a more detailed error output
wantDetailedOutput = False
#Change to True if you would like to output the full template database object
wantFullOutput = False

process.SiPixelTemplateDBObjectReader = cms.EDAnalyzer("SiPixelTemplateDBObjectReader",
                              siPixelTemplateCalibrationLocation = cms.string(
                             "CalibTracker/SiPixelESProducers"),
                              wantDetailedTemplateDBErrorOutput = cms.bool(wantDetailedOutput),
                              wantFullTemplateDBOutput = cms.bool(wantFullOutput))





####### DO ALL READERS (OR SELECT ONE YOU WANT) ########
process.p = cms.Path(
process.SiPixelCondObjOfflineReader*
process.SiPixelCondObjOfflineSimReader*
process.SiPixelLorentzAngleReader*
process.SiPixelLorentzAngleSimReader*
process.SiPixelFedCablingMapAnalyzer*
process.SiPixelCondObjForHLTReader*
process.SiPixelCondObjForHLTSimReader*
process.SiPixelTemplateDBObjectReader*
process.SiPixelBadModuleReader
)


