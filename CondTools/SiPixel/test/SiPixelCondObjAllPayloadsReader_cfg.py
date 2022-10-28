import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("PixelDBReader")

##
## prepare options
##
options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run3_data",VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('payloadType',
                  "HLT",VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "what payload type should be used")

options.register ('firstRun',
                  1,VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "first run to be processed")

options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo_GainsAll.root")
                                   )

# There is no actual payload in DB associated with the SiPixelGainCalibrationRcd record
# using the fake gain producer instead
process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGainESSource_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(10),
                            firstRun = cms.untracked.uint32(options.firstRun)
                            )

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(0)
)


process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
process.CondDB.DBParameters.messageLevel = 2
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDB,
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      toGet = cms.VPSet(#cms.PSet(record = cms.string('SiPixelGainCalibrationRcd'),
                                                        #         tag = cms.string('GainCalibTestFull')
                                                        #     ), 
                                                        cms.PSet(record = cms.string('SiPixelGainCalibrationForHLTRcd'),
                                                                 tag = cms.string('SiPixelGainCalibrationHLT_2009runs_express')
                                                             ), 
                                                        cms.PSet(record = cms.string('SiPixelGainCalibrationOfflineRcd'),
                                                                 tag = cms.string('SiPixelGainCalibration_2009runs_express')
                                                             ))
                                       )
process.prefer("PoolDBESSource")

process.SiPixelCondObjAllPayloadsReader = cms.EDAnalyzer("SiPixelCondObjAllPayloadsReader",
                                                         process.SiPixelGainCalibrationServiceParameters,
                                                         payloadType = cms.string(options.payloadType)
                                                         )

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjAllPayloadsReader)
#process.ep = cms.EndPath(process.print)

