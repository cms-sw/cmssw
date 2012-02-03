import FWCore.ParameterSet.Config as cms

l1TdeDTTPGClient = cms.EDAnalyzer("L1TdeDTTPGClient",
     runOnline   = cms.untracked.bool(True),
     hasBothThreshold  = cms.untracked.double(.85),
     qualThreshold     = cms.untracked.double(.90),
     phiThreshold  = cms.untracked.double(.85),
     phibendThreshold  = cms.untracked.double(.85),
     statQualThreshold = cms.untracked.double(.90)
)


