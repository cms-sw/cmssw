import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *
from HLTriggerOffline.Muon.muonTriggerRateTimeAnalyzer_cfi import *

HLTMuonVal = cms.Sequence(
    recoMuonValidationHLT_seq + 
    muonTriggerRateTimeAnalyzer
    )

HLTMuonVal_FastSim = cms.Sequence(
    muonTriggerRateTimeAnalyzer
    )

