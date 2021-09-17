import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStripQualityStatisticsSingleTag")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('tagName',
                  "NOTATAG",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "DB tag name")
options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    log_singletag = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
        SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
    ),
    destinations = cms.untracked.vstring('log_singletag','cout'),
    categories = cms.untracked.vstring('SiStripQualityStatistics')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue= cms.uint64(options.runNumber),
    lastValue= cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.Geometry.GeometryIdeal_cff")   # needed because the GlobalTag is NOT used
process.load("CondCore.DBCommon.CondDBCommon_cfi")   # needed because the GlobalTag is NOT used
process.CondDBCommon.connect='frontier://FrontierProd/CMS_CONDITIONS'
process.poolDBESSource=cms.ESSource("PoolDBESSource",
                                    process.CondDBCommon,
                                    BlobStreamerName=cms.untracked.string('TBufferBlobStreamingService'),
                                    toGet           =cms.VPSet(
    cms.PSet(
    record=cms.string('SiStripBadModuleRcd'),
    tag   =cms.string(options.tagName)
    )
    )
                                    )

# Include masking #

process.onlineSiStripQualityProducer = cms.ESProducer("SiStripQualityESProducer",
   appendToDataLabel = cms.string(''),
   PrintDebugOutput = cms.bool(False),
   PrintDebug = cms.untracked.bool(True),
   ListOfRecordToMerge = cms.VPSet(cms.PSet(
       record = cms.string('SiStripBadModuleRcd'),
       tag = cms.string('')
       )),
   UseEmptyRunInfo = cms.bool(False),
   ReduceGranularity = cms.bool(False),
#   ThresholdForReducedGranularity = cms.double(0.3)
)

#-------------------------------------------------
# Services for the TkHistoMap
#-------------------------------------------------
process.load("DQM.SiStripCommon.TkHistoMap_cff")

from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        TkMapFileName=cms.untracked.string("TkMapBadComponents_singleTag.png")  #available filetypes: .pdf .png .jpg .svg
        )

process.p = cms.Path(process.stat)


