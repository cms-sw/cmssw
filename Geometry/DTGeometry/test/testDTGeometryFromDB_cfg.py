import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBSetup,
                                      loadAll = cms.bool(True),
                                      toGet = cms.VPSet(cms.PSet(
                                                                 record =
                                                                 cms.string('DTRecoGeometryRcd'),
                                                                 tag =
                                                                 cms.string('XMLFILE_TEST_01')
                                                                )),
                                      DBParameters = cms.PSet(
                                                              messageLevel = cms.untracked.int32(9),
                                                              authenticationPath = cms.untracked.string('.')
                                                             ),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('sqlite_file:test.db')
                                     )


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
#process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_30X::All' 
#process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 
#process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "FakeAlignmentSource")
# process.fake2 = process.FakeAlignmentSource
# del process.FakeAlignmentSource
# process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "fake2")


#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
#process.e = cms.EndPath(process.out)


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 1000000
