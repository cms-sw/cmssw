import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
options.register("multirun", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

process = cms.Process('HARVESTING')

# read all the DQMIO files produced by the previous jobs
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(
        "file:DQM.root",
    )
)

# DQMStore service
process.load('DQMServices.Core.DQMStore_cfi')

# FastTimerService client
process.load('HLTrigger.Timer.fastTimerServiceClient_cfi')
process.fastTimerServiceClient.dqmPath = "HLT/TimerService"

# DQM file saver
process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.workflow = "/HLT/FastTimerService/All"

process.DQMFileSaverOutput = cms.EndPath( process.fastTimerServiceClient + process.dqmSaver )

if options.multirun:
    # multirun harvesting
    process.DQMStore.collateHistograms  = True
    process.dqmSaver.saveByRun      = -1
    process.dqmSaver.saveAtJobEnd   = True
    process.dqmSaver.forceRunNumber = 999999

