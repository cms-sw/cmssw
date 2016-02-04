import FWCore.ParameterSet.Config as cms

process = cms.Process("testReadMVAComputerCondDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

from CondCore.DBCommon.CondDBSetup_cfi import *

process.BTauGenericMVAJetTagComputerRcd = cms.ESSource("PoolDBESSource",
	CondDBSetup,
	timetype = cms.untracked.string('runnumber'),
	toGet = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('Foobar_tag')
	)),
	connect = cms.string('sqlite_file:FoobarDiscriminator.db'),
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

process.testReadMVAComputerCondDB = cms.EDAnalyzer("testReadMVAComputerCondDB")

process.p = cms.Path(process.testReadMVAComputerCondDB)
