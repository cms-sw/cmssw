import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessorHLT_cff import *
from HLTriggerOffline.Muon.HLTMuonPostProcessor_cfi import *

HLTMuonPostVal = cms.Sequence(
    recoMuonPostProcessorsHLT +
    HLTMuonPostProcessor
    )

HLTMuonPostVal_FastSim = cms.Sequence(
    recoMuonPostProcessorsHLTFastSim +
    HLTMuonPostProcessor
    )

