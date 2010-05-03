import FWCore.ParameterSet.Config as cms

hltQTExample = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('HLTriggerOffline/Common/data/HltQTExample.xml'),
    #QualityTestPrescaler = cms.untracked.int32(1)
    ##check verbose levels in http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DQMServices/Components/src/QualityTester.cc 
    #reportThreshold = cms.untracked.string('black'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndJob=cms.untracked.bool(True),
    testInEventloop=cms.untracked.bool(False),
    qtestOnEndLumi=cms.untracked.bool(False)
)
