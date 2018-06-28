import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
process = cms.Process('CTPPSDQM', eras.Run2_2018, stage2L1Trigger)

test = False

# global tag - conditions for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# event source
if not test:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
else:
  # for testing in lxplus
  process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      # 2018, 90m alignment run, run 318551, TOTEM1x data stream
      '/store/data/Run2018B/TOTEM10/RAW/v1/000/318/551/00000/6ABB0170-3878-E811-B0C2-FA163EB7360C.root'
    )
  )

  from DQM.Integration.config.dqmPythonTypes import *
  process.runType = RunType()
  process.runType.setRunType("pp_run")

  process.GlobalTag.globaltag = '101X_dataRun2_HLT_v7'

# DQM environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'CTPPS'
process.dqmSaver.tag = 'CTPPS'

if test:
  process.dqmSaver.path = "."

process.load("DQMServices.Components.DQMProvInfo_cfi")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

process.load("L1Trigger.Configuration.L1TRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# DQM Modules
process.load("DQM.CTPPS.ctppsDQM_cff")

# processing path
process.recoStep = cms.Sequence(
  process.L1TRawToDigi *
  process.ctppsRawToDigi *
  process.recoCTPPS
)

process.dqmModules = cms.Sequence(
  process.ctppsDQM +
  process.dqmEnv +
  process.dqmSaver
)

process.path = cms.Path(
  process.recoStep *
  process.dqmModules
)

process.schedule = cms.Schedule(process.path)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
