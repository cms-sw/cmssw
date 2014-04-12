import FWCore.ParameterSet.Config as cms

process = cms.Process("SVG")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Configuration.StandardSequences.Geometry_cff")
process.MuonGeometrySVGTemplate = cms.EDAnalyzer("MuonGeometrySVGTemplate",
                                                 wheelTemplateName = cms.string("wheel_template.svg"))

process.Path = cms.Path(process.MuonGeometrySVGTemplate)
