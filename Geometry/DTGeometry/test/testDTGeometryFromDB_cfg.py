import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration/StandardSequences/GeometryDB_cff")

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

# process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# process.GlobalTag.globaltag = 'IDEAL_31X::All' 

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
