import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/StandardSequences/GeometryDB_cff')
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_GEOM_RPCV2::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test1 = cms.EDFilter("RPCGEO")
process.test2 = cms.EDFilter("RPCGeometryAnalyzer")
process.demo = cms.EDAnalyzer("PrintEventSetupContent")
process.test3 = cms.EDFilter("ExtractXMLFile", label=cms.untracked.string("Extended"), fname=cms.untracked.string("fred2.xml"))

process.p = cms.Path(process.test1+process.test2+process.demo+process.test3)

