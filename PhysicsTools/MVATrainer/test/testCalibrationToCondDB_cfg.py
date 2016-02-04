import FWCore.ParameterSet.Config as cms

process = cms.Process("IPCalibrationToCondDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

process.MVAComputerESSource = cms.ESSource("MVAComputerESSource",
	Test = cms.string('TrainedGauss.mva')
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:TrainedGauss.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('MVAComputerRecord'),
		tag = cms.string('Test')
	))
)

process.MVAComputerSave = cms.EDFilter("MVAComputerTrainerSave",
	toPut = cms.vstring(),
	toCopy = cms.vstring('Test')
)

process.outpath = cms.EndPath(process.MVAComputerSave)
