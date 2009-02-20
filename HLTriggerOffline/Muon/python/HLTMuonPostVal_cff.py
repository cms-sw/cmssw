import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessorHLT_cff import *
from HLTriggerOffline.Muon.PostProcessor_cfi import *

HLTMuonPostVal = cms.Sequence(
    recoMuonPostProcessorsHLT +
    HLTMuonPostProcessor
    )
