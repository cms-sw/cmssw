import FWCore.ParameterSet.Config as cms
import socket
process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


# for live online DQM in P5
process.load("DQM.Integration.test.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

#process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'FED'
process.dqmSaver.dirName = '.'

process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")

# FED Integrity Client
process.load("DQMServices.Components.DQMFEDIntegrityClient_cff")
process.dqmFEDIntegrity.fedFolderName = cms.untracked.string("FEDIntegrity_EvF")

process.pDQM = cms.Path(process.l1tsClient+
                        process.hltsClient+
			process.dqmFEDIntegrityClient+
			process.dqmEnv+
			process.dqmSaver)


### process customizations included here
from DQM.Integration.test.online_customizations_cfi import *
process = customise(process)
