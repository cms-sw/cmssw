import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Geometry.TrackerGeometryBuilder.TrackerGeometryConstants_cff")

process.source = cms.Source("EmptySource")

#the following may not need the connect string if the global tag includes the extra.
#process.GlobalTag.toGet = cms.VPSet(
#    cms.PSet(record = cms.string("PGeometricDetExtraRcd"),
#             tag = cms.string("TKExtra_Geometry_38YV0"),
#             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY")
#             )
#    )

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


