import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelDBReader")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo_FullGains.root")
                                   )

process.CondDB.connect = 'sqlite_file:prova.db'
process.CondDB.DBParameters.messageLevel = 2
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

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

# There is no actual payload in DB associated with this record
# using the fake gain producer instead
process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGainESSource_cfi")

#process.load("CondCore.CondDB.CondDB_cfi")
# process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#     process.CondDB,
#     BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#     toGet = cms.VPSet(cms.PSet(
#         record = cms.string('SiPixelGainCalibrationRcd'),
#         tag = cms.string('GainCalib_v2_test')
#     ))
# )
#process.prefer("PoolDBESSource")

process.SiPixelCondObjReader = cms.EDAnalyzer("SiPixelCondObjReader",
    process.SiPixelGainCalibrationServiceParameters,
    maxRangeDeadPixHist = cms.untracked.double(0.001)
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjReader)
#process.ep = cms.EndPath(process.print)
