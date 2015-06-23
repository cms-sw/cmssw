import FWCore.ParameterSet.Config as cms

process = cms.Process("DDFilteredViewTest")

process.load('Configuration.Geometry.GeometryExtended2015_cff')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.fva = cms.EDAnalyzer("DDFilteredViewAnalyzer",
                             attribute = cms.string("OnlyForHcalSimNumbering"),
                             value = cms.string("HCAL"))

process.p1 = cms.Path(process.fva)

