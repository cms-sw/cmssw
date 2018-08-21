import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('CTPPSDQM', eras.Run2_2018)

test = False

# event source
if not test:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
else:
  # for testing in lxplus
  process.load("DQM.Integration.config.fileinputsource_cfi")
  process.source.fileNames = cms.untracked.vstring(
    #'root://eostotem.cern.ch//eos/totem/user/j/jkaspar/04C8034A-9626-E611-9B6E-02163E011F93.root'
    '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
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
