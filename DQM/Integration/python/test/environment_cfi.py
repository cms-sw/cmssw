import FWCore.ParameterSet.Config as cms

#from DQMServices.Core.DQM_cfg import *
from DQMServices.Core.DQMStore_cfi import *

DQM = cms.Service("DQM",                                                                                                                                                                           
                  debug = cms.untracked.bool(False),
                  publishFrequency = cms.untracked.double(5.0),
                  collectorPort = cms.untracked.int32(9090),
                  collectorHost = cms.untracked.string('dqm-prod-local.cms'),
                  filter = cms.untracked.string('')
                  )

DQMMonitoringService = cms.Service("DQMMonitoringService")

from DQMServices.Components.DQMEnvironment_cfi import *

dqmSaver.convention = 'Online'
dqmSaver.referenceHandling = 'all'
#dqmSaver.dirName = '/home/dqmprolocal/output'
dqmSaver.dirName = '.'
dqmSaver.producer = 'DQM'
dqmSaver.saveByLumiSection = 10
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = False

