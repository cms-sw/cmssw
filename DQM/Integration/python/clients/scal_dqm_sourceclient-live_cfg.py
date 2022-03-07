import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("DQM", Run3)

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
process.dqmEnv.subSystemFolder = 'Scal'
process.dqmSaver.tag = 'Scal'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'Scal'
process.dqmSaverPB.runNumber = options.runNumber
#-----------------------------
process.load("DQMServices.Components.DQMScalInfo_cfi")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi")

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()

if (process.runType.getRunType() == process.runType.pp_run and not unitTest):
    process.source.SelectEvents = ['HLT_ZeroBias*']

process.physicsBitSelector = cms.EDFilter("PhysDecl",
                                                   applyfilter = cms.untracked.bool(False),
                                                   debugOn     = cms.untracked.bool(False),
                                                   HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
                                          )


process.load("EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi")

## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
#process.load("Configuration.StandardSequences.Reconstruction_cff")

#-----------------------------
#### Sub-system configuration follows
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)
process.evfDQMmodulesPath = cms.Path(
                              process.l1GtUnpack*
			      process.gtDigis*
			      process.l1GtRecord*
			      process.physicsBitSelector*
                              process.scalersRawToDigi*
                              process.dqmscalInfo*
                              process.dqmmodules
                              )
process.schedule = cms.Schedule(process.evfDQMmodulesPath)

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = "rawDataRepacker"
    process.csctfDigis.producer = "rawDataRepacker"
    process.dttfDigis.DTTF_FED_Source = "rawDataRepacker"
    process.ecalDigis.cpu.InputLabel = "rawDataRepacker"
    process.ecalPreshowerDigis.sourceTag = "rawDataRepacker"
    process.gctDigis.inputLabel = "rawDataRepacker"
    process.gtDigis.DaqGtInputTag = "rawDataRepacker"
    process.gtEvmDigis.EvmGtInputTag = "rawDataRepacker"
    process.hcalDigis.InputLabel = "rawDataRepacker"
    process.muonCSCDigis.InputObjects = "rawDataRepacker"
    process.muonDTDigis.inputLabel = "rawDataRepacker"
    process.muonRPCDigis.InputLabel = "rawDataRepacker"
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.siPixelDigis.cpu.InputLabel = "rawDataRepacker"
    process.siStripDigis.ProductLabel = "rawDataRepacker"


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Final Source settings:", process.source)
