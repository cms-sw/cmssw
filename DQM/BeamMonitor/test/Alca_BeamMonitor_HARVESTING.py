import FWCore.ParameterSet.Config as cms

process = cms.Process("HARVESTING")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:AlcaBeamMonitorDQM_160410.root ',
    ),
    processingMode = cms.untracked.string('RunsAndLumis')
)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamMonitorClient = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
#process.load('Configuration.EventContent.EventContent_cff')

process.load("DQM.BeamMonitor.AlcaBeamMonitorClient_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc08.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/DQM/TkAlCalibration/ALCARECO'
process.dqmEnv.subSystemFolder = 'AlcaBeamMonitor'
process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True

#import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
#process.offlineBeamSpotForDQM = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.pp = cms.Path(process.EDMtoME*process.AlcaBeamMonitorClient+process.dqmSaver)
process.schedule = cms.Schedule(process.pp)
