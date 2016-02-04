import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonDataCertification_cfi import *

process = cms.Process("photonDataCertification")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMOffline.EGamma.photonDataCertification_cfi")




process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptySource"
)



process.dqmInfoEgamma = cms.EDAnalyzer("DQMEventInfo",
                                         subSystemFolder = cms.untracked.string('Egamma')
                                     )



process.DQMStore = cms.Service("DQMStore")





#process.p = cms.Path(process.dqmInfoEgamma*process.photonDataCertification)
process.p = cms.Path(process.photonDataCertification)
process.schedule = cms.Schedule(process.p)
