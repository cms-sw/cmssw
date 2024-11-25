import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("CompareGeometryTest",_PH2_ERA,dd4hep)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.DD4hep_TestMTDIdealGeometry = cms.untracked.PSet(
    # limit = cms.untracked.int32(-1)
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.DD4hep_TestMTDNumbering = cms.untracked.PSet(
    # limit = cms.untracked.int32(-1)
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.DD4hep_TestMTDPath = cms.untracked.PSet(
    # limit = cms.untracked.int32(-1)
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.DD4hep_TestMTDPosition = cms.untracked.PSet(
    # limit = cms.untracked.int32(-1)
    limit = cms.untracked.int32(0)
)
process.MessageLogger.files.mtdCommonDataDD4hep = cms.untracked.PSet(
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

process.load('Configuration.Geometry.GeometryDD4hepExtendedRun4Default_cff')

process.testBTL = cms.EDAnalyzer("DD4hep_TestMTDIdealGeometry",
                                 DDDetector = cms.ESInputTag('',''),
                                 ddTopNodeName = cms.untracked.string('BarrelTimingLayer')
                                )

process.testETL = cms.EDAnalyzer("DD4hep_TestMTDIdealGeometry",
                                 DDDetector = cms.ESInputTag('',''),
                                 ddTopNodeName = cms.untracked.string('EndcapTimingLayer')
                                )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(cms.wait(process.testBTL)+process.testETL)

