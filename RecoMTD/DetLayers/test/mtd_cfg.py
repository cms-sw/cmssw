import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("GeometryTest",_PH2_ERA,dd4hep)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.MTDLayerDumpFull = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.MTDDetLayersFull = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.MTDLayerDump = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.MTDDetLayers = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.files.mtdDetLayerGeometry = cms.untracked.PSet(
    MTDLayerDump = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    MTDDetLayers = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
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
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    noLineBreaks = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'))

# Choose Tracker Geometry
process.load("Configuration.Geometry.GeometryDD4hepExtendedRun4DefaultReco_cff")
process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("MTDRecoGeometryAnalyzer")
process.prod1 = cms.EDAnalyzer("TestBTLNavigation")
process.prod2 = cms.EDAnalyzer("TestETLNavigation")

# process.p1 = cms.Path(cms.wait(process.prod)+cms.wait(process.prod1)+process.prod2)
process.p1 = cms.Path(cms.wait(process.prod)+process.prod2)
