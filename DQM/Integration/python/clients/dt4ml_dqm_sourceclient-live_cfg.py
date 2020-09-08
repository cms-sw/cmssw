from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys
import os

process = cms.Process("DTDQM")

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

#----------------------------
#### Event Source
#----------------------------
if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    # for live online DQM in P5
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")

#----------------------------
#### DQM Live Environment
#----------------------------
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.tag = "DT"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "DT"
process.dqmSaverPB.runNumber = options.runNumber
#-----------------------------

### CUSTOMIZE FOR ML

# prepare the output directory
filePath = "/globalscratch/dqm4ml_" + process.dqmRunConfig.type.value()
if unitTest:
    filePath = "./dqm4ml_" + process.dqmRunConfig.type.value()
try:
    os.makedirs(filePath)
except:
    pass

process.dqmSaver.backupLumiCount = 10
process.dqmSaver.keepBackupLumi = True

process.dqmSaver.path = filePath
process.dqmSaverPB.path = filePath + "/pb"

# disable DQM gui
print("old:",process.DQM.collectorHost)
process.DQM.collectorHost = cms.untracked.string('dqm-blackhole.cms')
print("new:",process.DQM.collectorHost)
### END OF CUSTOMIZE FOR ML

# DT reco and DQM sequences
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
#---- for offline DB: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.physicsEventsFilter *  process.dtDQMPhysSequence)

#process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter * process.dtDQMCalib)

process.twinMuxStage2Digis.DTTM7_FED_Source = cms.InputTag("rawDataCollector")
process.dtunpacker.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")

print("Running with run type = ", process.runType.getRunType())

#----------------------------
#### pp run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.pp_run):
    pass


#----------------------------
#### cosmic run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.cosmic_run):
    pass


#----------------------------
#### HI run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.hi_run):
    process.twinMuxStage2Digis.DTTM7_FED_Source = cms.InputTag("rawDataRepacker")
    process.dtunpacker.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    
    process.dtDigiMonitor.ResetCycle = cms.untracked.int32(9999)



### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
