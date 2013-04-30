import os
import shlex, subprocess

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelGainBuilder")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

# phase1
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixel_cff')

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

try:
    user = os.environ["USER"]
except KeyError:
    user = subprocess.call('whoami')
    # user = commands.getoutput('whoami')
 
#file = "/tmp/" + user + "/prova.db"
file = "prova.db"
sqlfile = "sqlite_file:" + file
print '\n-> Uploading as user %s into file %s, i.e. %s\n' % (user, file, sqlfile)

#subprocess.call(["/bin/cp", "prova.db", file])
subprocess.call(["/bin/mv", "prova.db", "prova_old.db"])


##### DATABASE CONNNECTION AND INPUT TAGS ######
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(1),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(sqlfile),
    toPut = cms.VPSet(
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
        )
                     )
)







###### OFFLINE GAIN OBJECT ######
process.SiPixelCondObjOfflineBuilder = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    #numberOfModules = cms.int32(2000),
    numberOfModules = cms.int32(0),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.077),
    meanPed = cms.double(28.0),
    #meanPed = cms.double(-6.37),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.077),
    meanPedFPix = cms.untracked.double(28.0),
    #meanPedFPix = cms.untracked.double(-6.37),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjOfflineBuilderSim = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    #numberOfModules = cms.int32(2000),
    numberOfModules = cms.int32(0),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.077),
    meanPed = cms.double(28.0),
    #meanPed = cms.double(-6.37),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.077),
    meanPedFPix = cms.untracked.double(28.0),
    #meanPedFPix = cms.untracked.double(-6.37),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


##### HLT GAIN OBJECT #####
process.SiPixelCondObjForHLTBuilder = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    #numberOfModules = cms.int32(2000),
    numberOfModules = cms.int32(0),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    meanGain = cms.double(2.077),
    meanPed = cms.double(28.0),
    #meanPed = cms.double(-6.37),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjForHLTBuilderSim = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    #numberOfModules = cms.int32(2000),
    numberOfModules = cms.int32(0),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    rmsPed = cms.double(0.0),
    meanGain = cms.double(2.077),
    meanPed = cms.double(28.0),
    #meanPed = cms.double(-6.37),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


process.p = cms.Path(
#    process.SiPixelCondObjOfflineBuilder
#    process.SiPixelCondObjForHLTBuilder
#    process.SiPixelCondObjOfflineBuilderSim
    process.SiPixelCondObjForHLTBuilderSim
    )

