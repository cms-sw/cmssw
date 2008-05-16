import FWCore.ParameterSet.Config as cms

# the clients
from DQMOffline.Muon.trackResidualsTest_cfi import *
muonSourcesQualityTests = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests1.xml')
)

muonClientsQualityTests = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests2.xml')
)

muonQualityTests = cms.Sequence(muonSourcesQualityTests*muTrackResidualsTest*muonClientsQualityTests)

