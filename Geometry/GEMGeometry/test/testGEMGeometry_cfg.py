import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Configuration.Geometry.GeometryExtended2019_cff")
process.load("Configuration.Geometry.GeometryExtended2019Reco_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')


from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test2 = cms.EDAnalyzer("GEMGeometryAnalyzer")

#process.p = cms.Path(process.test1+process.test2)
process.p = cms.Path(process.test2)

