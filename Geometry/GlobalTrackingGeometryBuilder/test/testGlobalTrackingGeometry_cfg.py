import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("Demo",Phase2C17I13M9)
process.load('Configuration.Geometry.GeometryExtended2026D99Reco_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.GlobalTracking=dict()

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("GlobalTrackingGeometryTest")

process.p = cms.Path(process.test)

