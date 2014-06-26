import FWCore.ParameterSet.Config as cms
import socket
process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("DQMHttpSource",
               sourceURL =   cms.untracked.string('http://%s:22100/urn:xdaq-application:lid=30' % socket.gethostname()),        
               DQMconsumerName = cms.untracked.string('FED EventFilter Histogram Consumer'),
               DQMconsumerPriority = cms.untracked.string('normal'),
               headerRetryInterval = cms.untracked.int32(5),
               maxDQMEventRequestRate =  cms.untracked.double(1.0),
               topLevelFolderName =    cms.untracked.string('*')
                            )

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'FED'

process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")

# FED Integrity Client
process.load("DQMServices.Components.DQMFEDIntegrityClient_cff")

process.pDQM = cms.Path(process.l1tsClient+
                        process.hltsClient+
			process.dqmFEDIntegrityClient+
			process.dqmEnv+
			process.dqmSaver)
