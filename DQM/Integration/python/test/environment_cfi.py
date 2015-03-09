import FWCore.ParameterSet.Config as cms

#from DQMServices.Core.DQM_cfg import *
from DQMServices.Core.DQMStore_cfi import *

DQM = cms.Service("DQM",                                                                                                                                                                           
                  debug = cms.untracked.bool(False),
                  publishFrequency = cms.untracked.double(5.0),
                  collectorPort = cms.untracked.int32(9190),
                  collectorHost = cms.untracked.string('dqm-c2d07-29.cms'),
                  filter = cms.untracked.string('')
                  )

DQMMonitoringService = cms.Service("DQMMonitoringService")

from DQMServices.Components.DQMEnvironment_cfi import *

dqmSaver.convention = 'Online'
dqmSaver.referenceHandling = 'all'
dqmSaver.dirName = '.'
dqmSaver.producer = 'Playback'
dqmSaver.saveByLumiSection = 10
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = False

