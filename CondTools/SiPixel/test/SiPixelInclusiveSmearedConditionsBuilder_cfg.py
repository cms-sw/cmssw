import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveBuilder")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

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
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
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
#    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_PIXEL'),
                                          connect = cms.string('sqlite_file:prova.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelFedCablingMapRcd'),
        tag = cms.string('SiPixelFedCablingMap_v14')
    ), 
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle_v01')
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
            record = cms.string('SiPixelLorentzAngleSimRcd'),
            tag = cms.string('SiPixelLorentzAngle_v01')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_const')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
        ))
)

process.PixelToLNKAssociateFromAsciiESProducer = cms.ESProducer("PixelToLNKAssociateFromAsciiESProducer",
    fileName = cms.string('pixelToLNK.ascii')
)

process.MapWriter = cms.EDAnalyzer("SiPixelFedCablingMapWriter",
    record = cms.string('SiPixelFedCablingMapRcd'),
    associator = cms.untracked.string('PixelToLNKAssociateFromAscii')
)

process.SiPixelLorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8)
)
process.SiPixelLorentzAngleSim = cms.EDAnalyzer("SiPixelLorentzAngleDB",
                                              magneticField = cms.double(3.8),
                                              record=cms.untracked.string("SiPixelLorentzAngleSimRcd")
)

process.SiPixelCondObjOfflineBuilder = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.0),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.14),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(2.75),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0),
    noisyFraction = cms.double(0.0)
)

process.SiPixelCondObjOfflineSimBuilder = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.0),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0),
    noisyFraction = cms.double(0.0)                                                       
)

process.SiPixelCondObjForHLTBuilder = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.14),
    meanGain = cms.double(0.4),
    meanPed = cms.double(50.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(2.75),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0),
    deadFraction = cms.double(0.0),
    noisyFraction = cms.double(0.0)
)
process.SiPixelCondObjForHLTSimBuilder = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.14),
    meanGain = cms.double(0.4),
    meanPed = cms.double(50.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(2.75),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0),
    deadFraction = cms.double(0.0),
    noisyFraction = cms.double(0.0)                                                      
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjOfflineBuilder*process.SiPixelCondObjForHLTBuilder)
#process.ep = cms.EndPath(process.print)


