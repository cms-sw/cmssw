import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/StandardSequences/GeometryDB_cff')
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_37Y_V0::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

#process.test1 = cms.EDAnalyzer("RPCGEO")
#process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")
#process.demo = cms.EDAnalyzer("PrintEventSetupContent")
process.test3 = cms.EDAnalyzer("ExtractXMLFile", label=cms.untracked.string("Extended"), fname=cms.untracked.string("fred.xml"))

#process.p = cms.Path(process.test1+process.test2+process.demo+process.test3)
process.p = cms.Path(process.test3)

