import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")

#this is always needed if users want access to the vector<GeometricDetExtra>
process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                           fromDDD = cms.bool( False )
                                                           )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("ModuleInfo",
    fromDDD = cms.bool(False)
)

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)


