import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("GeometryTest",Phase2C17I13M9,dd4hep)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.MTDDigiGeometryAnalyzer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.DD4hep_TestBTLPixelTopology = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.files.mtdGeometryDD4hep = cms.untracked.PSet(
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    FWKINFO = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MTDUnitTest = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    noLineBreaks = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO')
)

process.load("Configuration.Geometry.GeometryDD4hepExtended2026D98Reco_cff")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("MTDDigiGeometryAnalyzer")
process.prod1 = cms.EDAnalyzer("DD4hep_TestBTLPixelTopology",
    DDDetector = cms.ESInputTag('',''),
)

process.p1 = cms.Path(process.prod+process.prod1)
