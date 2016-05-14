import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessorHLT_cff import *
# ADD new validation
from Validation.RecoMuon.NewPostProcessorHLT_cff import *
#
from HLTriggerOffline.Muon.hltMuonPostProcessors_cff import *

# to be customized for OLD or NEW validation
HLTMuonPostVal = cms.Sequence(
#    recoMuonPostProcessorsHLT +
    NEWrecoMuonPostProcessorsHLT +
#
    hltMuonPostProcessors
    )

