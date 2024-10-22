import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelDBReader")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo_GainsForHLT.root")
                                   )

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
process.CondDB.DBParameters.messageLevel = 2
process.CondDB.DBParameters.authenticationPath = ''

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

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationForHLTRcd'),
        tag = cms.string('SiPixelGainCalibrationHLT_2009runs_express')
    ))
)

process.prefer("PoolDBESSource")
process.SiPixelCondObjForHLTReader = cms.EDAnalyzer("SiPixelCondObjForHLTReader",
                                                    process.SiPixelGainCalibrationServiceParameters,
                                                    maxRangeDeadPixHist = cms.untracked.double(0.001),
                                                    useSimRcd = cms.bool(False)
                                                    )

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjForHLTReader)
#process.ep = cms.EndPath(process.print)
