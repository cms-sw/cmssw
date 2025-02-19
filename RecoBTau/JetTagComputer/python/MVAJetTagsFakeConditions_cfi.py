import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

BTauMVAJetTagComputerRecord = cms.ESSource("PoolDBESSource",
	CondDBSetup,
	timetype = cms.string('runnumber'),
	toGet = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('MVAJetTags_CMSSW_2_0_0_mc')
	)),
	connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/MVAJetTagsFakeConditions.db'),
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)
