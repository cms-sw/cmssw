import FWCore.ParameterSet.Config as cms

process = cms.Process("MakeCondDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:FoobarDiscriminator.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('Foobar_tag')
	))
)

process.MakeCondDB = cms.EDAnalyzer("testWriteMVAComputerCondDB",
	record = cms.untracked.string('BTauGenericMVAJetTagComputerRcd')
)

process.p = cms.Path(process.MakeCondDB)
