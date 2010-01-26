import FWCore.ParameterSet.Config as cms

# Copy example MVA training into a local SQLite file
# Adapted from PhysicsTools/MVATrainer/test/testWriteMVAComputerCondDB_cfg.py
# Original author: Christopher Saout
# Modifications by Evan Friis

# The files specified in the MVAComputerES source (i.e. ZTauTauTraining, can be more than one) will be added to an MVAComputerContainer
# This computer containiner will be added to the specified database (in this case, Example.db) with the 'tag' given (i.e. MyTestMVATag)
# The 'toCopy' parameter lists the computers in the computer container to put into the database

process = cms.Process("TauMVACondUpload")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )


process.MVAComputerESSource = cms.ESSource("TauMVAComputerESSource",
#syntax:
#       label                   = cms.string('[filename]') 
	ZTauTauTraining         = cms.string('Example.mva')
	,ZTauTauTrainingCopy2   = cms.string('Example.mva')  #normally you would put different trainings in
                                                             #an example could be files trained with/without isolation req.
)


process.MVAComputerSave = cms.EDAnalyzer("TauMVATrainerSave",
	toPut = cms.vstring(),
        #list of labels to add into the tag given in the PoolDBOutputService
	#toCopy = cms.vstring('ZTauTauTraining', 'ZTauTauTrainingCopy2')
	toCopy = cms.vstring('ZTauTauTraining')
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:Example.db'),  #or frontier, etc
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('TauTagMVAComputerRcd'),
		tag = cms.string('MyTestMVATag')                               
	))
)

process.outpath = cms.EndPath(process.MVAComputerSave)
