import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("test")

##
## prepare options
##
options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run3_data",VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,        # string, int, or float
                  "GlobalTag")

options.register ('forHLT',
                  True,VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "payload type to be used")

options.register ('firstRun',
                  1,VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "first run to be processed")

options.parseArguments()

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.insertNoisyandDead = cms.EDAnalyzer("SiPixelGainCalibrationRejectNoisyAndDead",
                                            record = cms.untracked.string('SiPixelGainCalibrationForHLTRcd' if(options.forHLT) else 'SiPixelGainCalibrationOfflineRcd'),
                                            debug = cms.untracked.bool(False)              
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(options.firstRun),
                            interval = cms.uint64(1)
                            )

#Input DB
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(10),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationForHLTRcd' if(options.forHLT) else 'SiPixelGainCalibrationOfflineRcd'),
        tag = cms.string('SiPixelGainCalibrationHLT_2009runs_express' if(options.forHLT) else 'SiPixelGainCalibration_2009runs_express')
    )),
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
)

#Output DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(10),
        authenticationPath = cms.untracked.string('.')
    ),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTRcd' if(options.forHLT) else 'SiPixelGainCalibrationOfflineRcd'),
            tag = cms.string('GainCalib_TEST_hlt' if(options.forHLT) else 'GainCalib_Offline_hlt')
    )),
    connect = cms.string('sqlite_file:SiPixelGainCalibrationRejectedNoisyAndDead.db')
)

process.prefer("PoolDBESSource")
process.p = cms.Path(process.insertNoisyandDead)
