from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("ESDQM", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("FWCore.Modules.preScaler_cfi")

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

# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

process.load("EventFilter.ESRawToDigi.esRawToDigi_cfi")
#process.ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
process.esRawToDigi.sourceTag = 'source'
process.esRawToDigi.debugMode = False

process.load('RecoLocalCalo/EcalRecProducers/ecalPreshowerRecHit_cfi')
process.ecalPreshowerRecHit.ESdigiCollection = "esRawToDigi"
process.ecalPreshowerRecHit.ESRecoAlgo = 0

process.preScaler.prescaleFactor = 1

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
#process.dqmInfoES = DQMEDAnalyzer('DQMEventInfo',
#                                   subSystemFolder = cms.untracked.string('EcalPreshower')
#                                   )

#process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'EcalPreshower'
process.dqmSaver.tag = 'EcalPreshower'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'EcalPreshower'
process.dqmSaverPB.runNumber = options.runNumber
# for local test
#process.dqmSaver.path = '.'
#process.dqmSaverPB.path = './pb'

process.load("DQM/EcalPreshowerMonitorModule/EcalPreshowerMonitorTasks_cfi")
process.ecalPreshowerIntegrityTask.ESDCCCollections = "esRawToDigi"
process.ecalPreshowerIntegrityTask.ESKChipCollections = "esRawToDigi"
process.ecalPreshowerIntegrityTask.ESDCCCollections = "esRawToDigi"
process.ecalPreshowerIntegrityTask.ESKChipCollections = "esRawToDigi"
process.ecalPreshowerOccupancyTask.DigiLabel = "esRawToDigi"
process.ecalPreshowerPedestalTask.DigiLabel = "esRawToDigi"
process.ecalPreshowerRawDataTask.ESDCCCollections = "esRawToDigi"
process.ecalPreshowerTimingTask.DigiLabel = "esRawToDigi"
process.ecalPreshowerTrendTask.ESDCCCollections = "esRawToDigi"

process.load("DQM/EcalPreshowerMonitorClient/EcalPreshowerMonitorClient_cfi")
del process.dqmInfoES
process.p = cms.Path(process.preScaler*
               process.esRawToDigi*
               process.ecalPreshowerRecHit*
               process.ecalPreshowerDefaultTasksSequence*
               process.dqmEnv*
               process.ecalPreshowerMonitorClient*
               process.dqmSaver*
               process.dqmSaverPB)


process.esRawToDigi.sourceTag = "rawDataCollector"
process.ecalPreshowerRawDataTask.FEDRawDataCollection = "rawDataCollector"
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())

if (process.runType.getRunType() == process.runType.hi_run):
    process.esRawToDigi.sourceTag = "rawDataRepacker"
    process.ecalPreshowerRawDataTask.FEDRawDataCollection = "rawDataRepacker"


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
print("Final Source settings:", process.source)
process = customise(process)
