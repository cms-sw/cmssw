import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

test = False

# event source
if not test:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
else:
  # for testing in lxplus
  process.load("DQM.Integration.config.fileinputsource_cfi")
  process.source.fileNames = cms.untracked.vstring(
    "'file:///afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root"
  )


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

# global tag - conditions for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# raw-to-digi conversion
process.load("EventFilter.TotemRawToDigi.totemRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# DQM Modules
process.load("DQM.CTPPS.totemDQM_cff")

# processing path
process.recoStep = cms.Sequence(
  process.totemTriggerRawToDigi *
  process.totemRPRawToDigi *
  process.recoCTPPS
)

process.dqmModules = cms.Sequence(
  process.totemDQM +
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
