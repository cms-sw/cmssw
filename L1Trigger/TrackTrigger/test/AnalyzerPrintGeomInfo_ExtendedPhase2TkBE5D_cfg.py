import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("GeometrySummary")

process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MyModule = cms.EDAnalyzer("AnalyzerPrintGeomInfo",
    TextOutput = cms.string('GeometrySummary_ExtendedPhase2TkBE5D.log'),
    DebugMode = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('GeometrySummary_ExtendedPhase2TkBE5D.root'),
)

process.p1 = cms.Path(process.MyModule)
process.schedule = cms.Schedule(process.p1)

