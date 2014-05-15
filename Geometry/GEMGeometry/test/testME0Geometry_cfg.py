import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load("Configuration.Geometry.GeometryExtended2023HGCal_cff")
process.load("Configuration.Geometry.GeometryExtended2023HGCalReco_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('ME0GeometryBuilderFromDDD')
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"))


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test = cms.EDAnalyzer("ME0GeometryAnalyzer")

process.p = cms.Path(process.test)

