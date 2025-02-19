import FWCore.ParameterSet.Config as cms

hcalDAQInfo = cms.EDAnalyzer('HcalDAQInfo',
                             debug = cms.untracked.int32(0),
                             subSystemFolder = cms.untracked.string("Hcal")
)
