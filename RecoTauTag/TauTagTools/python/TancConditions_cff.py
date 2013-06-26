import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

TauTagMVAComputerRecord = cms.ESSource("PoolDBESSource",
	CondDBSetup,
	timetype = cms.string('runnumber'),
	toGet = cms.VPSet(cms.PSet(
		record = cms.string('TauTagMVAComputerRcd'),
		tag = cms.string('TauNeuralClassifier')
	)),
        connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/RecoTauTagConditions_31X.db'),
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

