import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessorHLT_cff import *
from HLTriggerOffline.Muon.hltMuonPostProcessors_cff import *

HLTMuonPostVal = cms.Sequence(
    recoMuonPostProcessorsHLT +
    hltMuonPostProcessors
    )

HLTMuonPostVal_FastSim = cms.Sequence(
    recoMuonPostProcessorsHLTFastSim +
    hltMuonPostProcessors
    )

