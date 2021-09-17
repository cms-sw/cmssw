import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DQM")

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

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

# Global tag - Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'Info'
process.dqmSaver.tag = 'Info'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'Info'
process.dqmSaverPB.runNumber = options.runNumber
#-----------------------------

# Digitisation: produce the Scalers digis containing DCS bits
process.load("EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi")
# Digitisation: produce the TCDS digis containing BST record
from EventFilter.Utilities.tcdsRawToDigi_cfi import *
process.tcdsDigis = tcdsRawToDigi.clone()

# OnlineMetaDataRawToDigi will put DCSRecord to an event
process.load('EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi')
process.onlineMetaDataDigis = cms.EDProducer('OnlineMetaDataRawToDigi')

# DQMProvInfo is the DQM module to be run
process.load("DQMServices.Components.DQMProvInfo_cfi")

# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)
process.evfDQMmodulesPath = cms.Path(
                                     process.scalersRawToDigi*
                                     process.tcdsDigis*
                                     process.onlineMetaDataRawToDigi*
                                     process.dqmProvInfo*
                                     process.dqmmodules
                                     )
process.schedule = cms.Schedule(process.evfDQMmodulesPath)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Heavy Ion Specific Fed Raw Data Collection Label
if (process.runType.getRunType() == process.runType.hi_run):
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.tcdsDigis.InputLabel = "rawDataRepacker"
else:
    process.scalersRawToDigi.scalersInputTag = "rawDataCollector"
    process.tcdsDigis.InputLabel = "rawDataCollector"

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
