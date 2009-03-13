import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveBuilder")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)



##### DATABASE CONNNECTION AND INPUT TAGS ######
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(3),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:test.db'),
    toPut = cms.VPSet(cms.PSet(
            record = cms.string('SiPixelFedCablingMapRcd'),
            tag = cms.string('SiPixelFedCablingMap_v14')
        ), 
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle_v01')
        ),
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleSimRcd'),
            tag = cms.string('SiPixelLorentzAngleSim_v01')
        ),
        cms.PSet(
            record = cms.string('SiPixelTemplateDBObjectRcd'),
            tag = cms.string('SiPixelTemplateDBObject')
        ),
        cms.PSet(
           record = cms.string('SiPixelQualityRcd'),
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
            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_const')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_hlt_const')
        ))
)








###### TEMPLATE OBJECT UPLOADER ######
process.TemplateUploader = cms.EDAnalyzer("SiPixelTemplateDBObjectUploader",
                                  siPixelTemplateCalibrations = cms.vstring(
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0001.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0004.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0011.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0012.out"),
                                  Version = cms.double("1.3")
)



###### QUALITY OBJECT MAKER #######
process.QualityObjectMaker = cms.EDFilter("SiPixelBadModuleByHandBuilder",
    BadModuleList = cms.untracked.VPSet(cms.PSet(
        errortype = cms.string('whole'),
        detid = cms.uint32(302197784)
         ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302195232)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344014348)
        )),
    Record = cms.string('SiPixelQualityRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    printDebug = cms.untracked.bool(True),
    doStoreOnDB = cms.bool(True)

)



##### CABLE MAP OBJECT ######
process.PixelToLNKAssociateFromAsciiESProducer = cms.ESProducer("PixelToLNKAssociateFromAsciiESProducer",
    fileName = cms.string('pixelToLNK.ascii')
)


process.MapWriter = cms.EDFilter("SiPixelFedCablingMapWriter",
    record = cms.string('SiPixelFedCablingMapRcd'),
    associator = cms.untracked.string('PixelToLNKAssociateFromAscii')
)



###### LORENTZ ANGLE OBJECT ######
process.SiPixelLorentzAngle = cms.EDFilter("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8),
    bPixLorentzAnglePerTesla = cms.double(0.106),
    fPixLorentzAnglePerTesla = cms.double(0.091),
    #in case lorentz angle values for bpix should be read from file -> not implemented yet
    useFile = cms.bool(False),
    record = cms.untracked.string('SiPixelLorentzAngleRcd'),  
    fileName = cms.string('lorentzFit.txt')	
)

process.SiPixelLorentzAngleSim = cms.EDFilter("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8),
    bPixLorentzAnglePerTesla = cms.double(0.106),
    fPixLorentzAnglePerTesla = cms.double(0.091),
    #in case lorentz angle values for bpix should be read from file -> not implemented yet
    useFile = cms.bool(False),
    record = cms.untracked.string('SiPixelLorentzAngleSimRcd'),
    fileName = cms.string('lorentzFit.txt')	
)

###### OFFLINE GAIN OBJECT ######
process.SiPixelCondObjOfflineBuilder = cms.EDFilter("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjOfflineBuilderSim = cms.EDFilter("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


##### HLT GAIN OBJECT #####
process.SiPixelCondObjForHLTBuilder = cms.EDFilter("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjForHLTBuilderSim = cms.EDFilter("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    rmsPed = cms.double(0.0),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


process.p = cms.Path(process.SiPixelLorentzAngle*process.MapWriter*process.SiPixelCondObjOfflineBuilder*process.SiPixelCondObjForHLTBuilder*process.TemplateUploader*process.QualityObjectMaker*
                     process.SiPixelLorentzAngleSim*process.SiPixelCondObjForHLTBuilderSim*process.SiPixelCondObjOfflineBuilderSim  )

