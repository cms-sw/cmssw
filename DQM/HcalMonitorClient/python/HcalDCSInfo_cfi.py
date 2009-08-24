import FWCore.ParameterSet.Config as cms

hcalDCSInfo = cms.EDAnalyzer('HcalDCSInfo',
                             debug = cms.untracked.int32(0),
                             subSystemFolder = cms.untracked.string("Hcal")
)
