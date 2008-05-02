import FWCore.ParameterSet.Config as cms

# the clients
from DQMOffline.Muon.trackResidualsTest_cfi import *
muonQualityTestsXML = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests.xml')
)

muonQualityTests = cms.Sequence(muTrackResidualsTest*muonQualityTestsXML)

