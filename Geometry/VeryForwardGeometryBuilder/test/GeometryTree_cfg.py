import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTree")

# logs
process.load("FWCore.MessageService.MessageLogger_cfi")

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")

# (no) events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

# teh analyzer
process.tree = cms.EDAnalyzer('GeometryTree')

process.p = cms.Path(process.tree)
