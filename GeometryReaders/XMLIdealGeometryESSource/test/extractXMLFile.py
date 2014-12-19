import FWCore.ParameterSet.Config as cms

process = cms.Process("ExtractXMLFile")

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test = cms.EDAnalyzer("ExtractXMLFile",
                               label=cms.untracked.string('Extended'),
                               fname=cms.untracked.string('fred.xml'))

process.p = cms.Path(process.test)

