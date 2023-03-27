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
    "/store/group/dpg_ctpps/comm_ctpps/PixelRandomTrigger2023/outputExpressPPSRandom.root"
  )
  process.source.inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
  
process.source.streamLabel = "streamDQMPPSRandom"

# DQM environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'PPSRANDOM'
process.dqmSaver.tag = 'PPSRANDOM'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'PPSRANDOM'
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
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import ctppsPixelDigis as _ctppsPixelDigis
process.ctppsPixelDigisAlCaRecoProducer = _ctppsPixelDigis.clone(inputLabel = 'hltPPSCalibrationRaw')

# loading Meta tags used by commonDQM
process.load('EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi')
process.onlineMetaDataDigis = cms.EDProducer('OnlineMetaDataRawToDigi')


# DQM Modules
process.load("DQM.CTPPS.ctppsDQM_cff")

# processing path
process.recoStep = cms.Sequence(
  process.ctppsPixelDigisAlCaRecoProducer *
  process.onlineMetaDataDigis
)

process.dqmModules = cms.Sequence(
  process.ctppsDQMRandomSource *
  process.ctppsDQMRandomHarvest
)

process.path = cms.Path(
  process.recoStep *
  process.dqmModules *

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
