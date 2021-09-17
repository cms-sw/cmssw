from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5
process.load("DQM.DTMonitorModule.test.inputsource_live_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmRunConfig.collectorHost = 'fu-c2f11-23-01.cms'
process.dqmSaver.path = "./"

#----------------------------
#### DQM Live Environment
#----------------------------
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.tag = "DT"
#-----------------------------

# Enable HLT*Mu* filtering to monitor on Muon events
# OR HLT_Physics* to monitor FEDs in commissioning runs
# process.source.SelectEvents = cms.untracked.vstring("HLT*Mu*","HLT_*Physics*")

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

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

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



### process customisations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

### DT slice test specific customisations
if (process.dtDqmConfig.getProcessAB7Digis() or \
    process.dtDqmConfig.getProcessAB7TPs()) :
    from DQM.DTMonitorModule.slice_test_customizations_cff import *
    process = customise_for_slice_test(process,
                                       process.dtDqmConfig.getProcessAB7Digis(),
                                       process.dtDqmConfig.getProcessAB7TPs())

### DT digi customisation
if (process.dtDqmConfig.getRunWithLargeTB()) :
    process.dtDigiMonitor.maxTTMounts = 6400

if (process.dtDqmConfig.getProcessAB7Digis()) :
    process.dtAB7DigiMonitor.maxTTMounts = 6400
    process.dtAB7DigiMonitor.tdcPedestal = process.dtDqmConfig.getTBTDCPedestal()
