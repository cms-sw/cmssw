import FWCore.ParameterSet.Config as cms

process = cms.Process("DTFineDelay")

# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")

# Geometry & Calibration Tag
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_P_V1::All"

# Produce fake TTCrx_delay = 0
process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FEF325E7-4E8B-DE11-BBC3-001D09F2AD4D.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FCC9B2CC-4E8B-DE11-8DB5-001D09F29533.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FCB064D8-688B-DE11-BCC3-001617C3B77C.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FCA03EBD-5A8B-DE11-9B50-000423D9880C.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FC8CE7D4-4E8B-DE11-B161-000423D98950.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FC1B590C-4E8B-DE11-8CED-001617DBD556.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FAC26775-4F8B-DE11-8812-000423D6CA6E.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FAA0AAF6-578B-DE11-B827-001D09F23A20.root'
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FA89F5EA-578B-DE11-9758-001617DC1F70.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/110/998/FA43BDA4-588B-DE11-A2FD-003048D37456.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.load("DQMServices.Core.DQM_cfg")

################# BEGIN DQM Online Environment #######################
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = "Online"
process.dqmSaver.dirName = "."
process.dqmSaver.producer = "DQM"
process.dqmSaver.saveByRun         =  1
process.dqmSaver.saveAtJobEnd      = True

process.dqmEnv.subSystemFolder = "DT"

process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
process.load("DQM.DTMonitorModule.dtTriggerSynchTask_cfi")
process.dtTriggerSynchMonitor.rangeWithinBX = False

# Load Fine Delay Correction Module
process.load("DQM.DTMonitorClient.dtFineDelayCorr_cfi")
process.dtFineDelayCorr.readOldFromDb = cms.bool(False)
process.dtFineDelayCorr.oldDelaysInputFile = cms.string("dtOldFineDelays.txt")
process.dtFineDelayCorr.writeDB = cms.bool(False)
process.dtFineDelayCorr.outputFile = cms.string("dtFineDelaysNew.txt")
process.dtFineDelayCorr.t0MeanHistoTag  = cms.string("TrackCrossingTimeAll")
process.dtFineDelayCorr.minEntries = cms.untracked.int32(100)


# Read worstPhases from DB
from CondCore.DBCommon.CondDBSetup_cfi import *
process.ttrigsource = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(record = cms.string('DTTPGParametersRcd'),
                               tag = cms.string('worstPhase')
                               )
                      ),
    connect = cms.string('sqlite_file:/afs/cern.ch/user/m/marinag/w0/finesync/cmssw/CMSSW_3_2_7/src/DQM/DTMonitorClient/test/worst_phase_map_112227.db'),
    authenticationMethod = cms.untracked.uint32(0)
    )

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTFineDelayCorr'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(-1)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(-1)),
                                                              DTFineDelayCorr = cms.untracked.PSet(
                                                                                  limit = cms.untracked.int32(-1))
                                                              )
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)
process.dtDQMPathPhys = cms.Path(process.dtTriggerSynchMonitor + process.dtFineDelayCorr + process.dqmmodules )
