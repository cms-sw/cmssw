import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

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
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/dt_reference.root'
#process.DQMStore.referenceFileName = "DT_reference.root"

#----------------------------
#### DQM Live Environment
#----------------------------
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.tag = "DT"
#-----------------------------


# DT reco and DQM sequences
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff") 
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
#---- for offline DB
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.physicsEventsFilter *  process.dtDQMPhysSequence)

#process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter * process.dtDQMCalib)

process.dttfunpacker.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.dtunpacker.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")

print "Running with run type = ", process.runType.getRunType()

#----------------------------
#### pp run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.pp_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/dt_reference_pp.root'


#----------------------------
#### cosmic run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.cosmic_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/dt_reference_cosmic.root'


#----------------------------
#### HI run settings 
#----------------------------

if (process.runType.getRunType() == process.runType.hi_run):
    process.dtunpacker.fedbyType = cms.bool(False)
    process.dttfunpacker.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.dtunpacker.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    
    process.dtDigiMonitor.ResetCycle = cms.untracked.int32(9999)

    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/dt_reference_hi.root'


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
