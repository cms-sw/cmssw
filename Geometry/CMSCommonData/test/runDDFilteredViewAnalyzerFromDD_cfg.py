import FWCore.ParameterSet.Config as cms

process = cms.Process("DDFilteredViewTest")

process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.CMSGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.fva = cms.EDAnalyzer("DDFilteredViewAnalyzer",
                             attribute = cms.string("OnlyForHcalSimNumbering"),
                             value = cms.string("HCAL"))

process.p1 = cms.Path(process.fva)

