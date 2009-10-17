import FWCore.ParameterSet.Config as cms

#from HLTriggerOffline.Common.HLTValidationQTExample_cfi import *
from HLTriggerOffline.Muon.HLTMuonQualityTester_cfi import *

hltvalidationqt = cms.Sequence(
    #hltQTExample
    hltMuonQualityTester
    )
