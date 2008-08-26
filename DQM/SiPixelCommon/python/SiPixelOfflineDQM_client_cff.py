import FWCore.ParameterSet.Config as cms

# DQM Services:
from DQMServices.Components.DQMEnvironment_cfi import *
dqmSaver.convention = 'Offline'
dqmSaver.dirName = '.'
dqmSaver.producer = 'DQM'
dqmEnv.subSystemFolder = 'Pixel'
dqmSaver.saveByLumiSection = -1
dqmSaver.saveByRun = 1
dqmSaver.saveAtJobEnd = True

sipixelEDAClient = cms.EDFilter("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True)
)

PixelOfflineDQMClient = cms.Sequence(sipixelEDAClient*dqmEnv*dqmSaver)
