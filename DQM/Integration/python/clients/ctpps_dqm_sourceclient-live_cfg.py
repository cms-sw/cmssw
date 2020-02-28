import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process('CTPPSDQM', Run2_2018)

test = False

# event source
if not test:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
else:
  # for testing in lxplus
  process.load("DQM.Integration.config.fileinputsource_cfi")
  process.source.fileNames = cms.untracked.vstring(
    #"root://eoscms.cern.ch//eos/cms/store/group/phys_pps/sw_test_input/001D08EE-C4B1-E711-B92D-02163E013864.root"
    #"/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root"
    "/store/data/Run2017B/SingleMuon/RAW/v1/000/297/050/00000/30346DF0-0153-E711-BBC7-02163E01437C.root"
  )
  process.source.inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
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
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# loading Meta tags used by commonDQM
process.load('EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi')
process.onlineMetaDataDigis = cms.EDProducer('OnlineMetaDataRawToDigi')


# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# DQM Modules
process.load("DQM.CTPPS.ctppsDQM_cff")

# processing path
process.recoStep = cms.Sequence(
  process.ctppsRawToDigi *
  process.onlineMetaDataDigis *
  process.recoCTPPS
)

process.dqmModules = cms.Sequence(
  process.ctppsDQMOnlineSource +
  process.ctppsDQMOnlineHarvest
)

process.dqmModulesCalibration = cms.Sequence(
  process.ctppsDQMCalibrationSource +
  process.ctppsDQMCalibrationHarvest
)

process.path = cms.Path(
  process.recoStep *

  # here: (un)comment to switch between normal and calibration mode
  process.dqmModules *
  #process.dqmModulesCalibration *

  process.dqmEnv *
  process.dqmSaver
)

process.schedule = cms.Schedule(process.path)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
