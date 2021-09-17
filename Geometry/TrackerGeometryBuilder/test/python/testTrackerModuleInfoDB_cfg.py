import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")


process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
                                                      fromDDD = cms.bool( False )
                                                     )

process.es_prefer_geomdet = cms.ESPrefer("TrackerGeometricDetESModule","")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("ModuleInfo",
    fromDDD = cms.bool(False),
    tolerance = cms.untracked.double(1.0e-23)
)

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)


