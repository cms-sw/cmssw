import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

#----------------------------
#### Event Source
#-----------------------------
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

#----------------------------
#### DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'DAQ'

# DAQ DQM Event Info program
    # code in DQMServices/XdaqCollector/src/XmasToEventInfo.cc
process.xmasdqmInfo= cms.EDFilter("XmasToDQMEventInfo",     
                  subSystemFolder = cms.untracked.string('DAQ'),              
                  prescaleEvt = cms.untracked.int32(-1)
                                  )

# DQM Source program
    # code in DQMServices/XdaqCollector/src/XmasToDQMSource.cc
process.xmasdqmSource= cms.EDFilter("XmasToDQMSource",
                  # Base name for monitor folders and output file      
                  monitorName = cms.untracked.string('DAQ'),
                  # Operate every N events (default: -1 no prescale)              
                  prescaleEvt = cms.untracked.int32(-1)
                                  )

### FIX YOUR  PATH TO INCLUDE
process.p = cms.Path(process.xmasdqmInfo*process.xmasdqmSource)
