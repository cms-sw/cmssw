import FWCore.ParameterSet.Config as cms

#from HLTriggerOffline.Common.HLTValidationQTExample_cfi import *
from HLTriggerOffline.Muon.HLTMuonQualityTester_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauQualityTests_cff import *
from HLTriggerOffline.Top.HLTTopQualityTester_cfi import *
from HLTriggerOffline.Higgs.HLTHiggsQualityTester_cfi import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETQualityTester_cfi import *
from HLTriggerOffline.SUSYBSM.HLTSusyExoQualityTester_cfi import *

hltvalidationqt = cms.Sequence(
    #hltQTExample
    hltMuonQualityTester
    + hltTauRelvalQualityTests
    + hltHiggsQualityTester
    + hltTopQualityTester
    + hltJetMetQualityTester
    + hltSusyExoQualityTester
    )
