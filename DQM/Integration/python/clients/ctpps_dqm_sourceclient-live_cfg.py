import FWCore.ParameterSet.Config as cms

import sys
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('CTPPSDQM', Run3)

test = False
unitTest = False

if 'unitTest=True' in sys.argv:
  unitTest=True

# event source
if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
elif not test:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
else:
  # for testing in lxplus
  process.load("DQM.Integration.config.fileinputsource_cfi")
  from DQM.Integration.config.fileinputsource_cfi import options
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
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'CTPPS'
process.dqmSaverPB.runNumber = options.runNumber

if test:
  process.dqmSaver.path = "."
  process.dqmSaverPB.path = "./pb"

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
process.load("RecoPPS.Configuration.recoCTPPS_cff")

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
  process.dqmSaver *
  process.dqmSaverPB
)

process.schedule = cms.Schedule(process.path)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
print("Final Source settings:", process.source)
process = customise(process)
