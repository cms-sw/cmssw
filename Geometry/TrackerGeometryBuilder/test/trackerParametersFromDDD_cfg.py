import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("TrackerParametersTest")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.Geometry.GeometryExtendedReco_cff')

process.trackerGeometry.applyAlignment = cms.bool(False)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.p1 = cms.Path(process.test)



