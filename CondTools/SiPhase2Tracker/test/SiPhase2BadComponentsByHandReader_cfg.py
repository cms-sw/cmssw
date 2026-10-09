import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('outputName', 'BadComponentsByHand_v0.db',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'sqlite file to read')
options.parseArguments()


process = cms.Process("READ")

process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2BadComponentsByHandReader = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1), reportEvery = cms.untracked.int32(1000)),
    SiPhase2BadComponentsByHandReader = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:' + options.outputName

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(
        cms.PSet(record = cms.string('SiPhase2OuterTrackerBadModuleRcd'), tag = cms.string('Phase2OTBadComponentsByHand_T35_v0')),
        cms.PSet(record = cms.string('SiPhase2InnerTrackerBadModuleRcd'), tag = cms.string('Phase2ITBadComponentsByHand_T35_v0')),
    )
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(
        cms.PSet(record = cms.string('SiPhase2OuterTrackerBadModuleRcd'), data = cms.vstring('SiPixelQuality')),
        cms.PSet(record = cms.string('SiPhase2InnerTrackerBadModuleRcd'), data = cms.vstring('SiPixelQuality')),
    ),
    verbose = cms.untracked.bool(True)
)

process.otReader = cms.EDAnalyzer("SiPhase2OTBadComponentsReader", printDebug = cms.untracked.bool(True))
process.itReader = cms.EDAnalyzer("SiPhase2ITBadComponentsReader", printDebug = cms.untracked.bool(True))

process.p = cms.Path(process.get + process.otReader + process.itReader)
