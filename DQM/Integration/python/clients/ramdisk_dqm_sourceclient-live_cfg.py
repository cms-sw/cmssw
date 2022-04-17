import FWCore.ParameterSet.Config as cms
import sys

subsystem = "Ramdisk"
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process(subsystem, Run3)

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process.load('DQM.Integration.config.inputsource_cfi')
from DQM.Integration.config.inputsource_cfi import options
process.load('DQMServices.Components.DQMEnvironment_cfi')
process.load('DQM.Integration.config.environment_cfi')

process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = subsystem
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = subsystem
process.dqmSaverPB.runNumber = options.runNumber

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.analyzer = DQMEDAnalyzer('RamdiskMonitor',
    runNumber = process.source.runNumber,
    runInputDir = process.source.runInputDir,
    streamLabels = cms.untracked.vstring(
        "streamDQM",
        "streamDQMHistograms",
        "streamDQMCalibration",
    )
)

process.p = cms.Path(process.analyzer)
process.dqmsave_step = cms.Path(process.dqmEnv * process.dqmSaver)

process.schedule = cms.Schedule(
    process.p,
    process.dqmsave_step
)
