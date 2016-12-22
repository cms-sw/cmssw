import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

# Global tag - Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'Info'
process.dqmSaver.tag = 'Info'
#-----------------------------

# Digitisation: produce the Scalers digis containing DCS bits
process.load("EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi")

# DQMProvInfo is the DQM module to be run
process.load("DQMServices.Components.DQMProvInfo_cfi")

# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)
process.evfDQMmodulesPath = cms.Path(
                                     process.scalersRawToDigi*
                                     process.dqmProvInfo*
                                     process.dqmmodules
                                     )
process.schedule = cms.Schedule(process.evfDQMmodulesPath)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Heavy Ion Specific Fed Raw Data Collection Label
if (process.runType.getRunType() == process.runType.hi_run):
    process.dqmProvInfo.fedRawData = cms.untracked.string("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
else:
    process.dqmProvInfo.fedRawData = cms.untracked.string("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
