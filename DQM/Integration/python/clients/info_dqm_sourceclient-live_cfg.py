import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'Info'
process.dqmSaver.tag = 'Info'
#-----------------------------
process.load("DQMServices.Components.DQMProvInfo_cfi")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag - Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Collision Reconstruction
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

##process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi")
##import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
##process.gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()

process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi")
import EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi
process.conditionsInEdm = EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi.conditionDumperInEdm.clone()

process.physicsBitSelector = cms.EDFilter("PhysDecl",
                                                   applyfilter = cms.untracked.bool(False),
                                                   debugOn     = cms.untracked.bool(False),
                                                   HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
                                          )

process.load("EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi")

process.dump = cms.EDAnalyzer('EventContentAnalyzer')

# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)
process.evfDQMmodulesPath = cms.Path(
                              process.l1GtUnpack*
			      process.gtDigis*
			      ##process.gtEvmDigis*
			      process.conditionsInEdm*
			      process.l1GtRecord*
			      process.physicsBitSelector*
                              process.scalersRawToDigi*
                              process.dqmProvInfo*
                              process.dqmmodules
)
process.schedule = cms.Schedule(process.evfDQMmodulesPath)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

# Heavy Ion Specific Fed Raw Data Collection Label
if (process.runType.getRunType() == process.runType.hi_run):
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    ##process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
else:
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
    ##process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")

# Process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
