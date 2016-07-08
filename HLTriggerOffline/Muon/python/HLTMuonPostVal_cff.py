import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessorHLT_cff import *
# add new muon validation
#from Validation.RecoMuon.NewPostProcessorHLT_cff import *
#
from HLTriggerOffline.Muon.hltMuonPostProcessors_cff import *

HLTMuonPostVal = cms.Sequence(
# to be customized for OLD or NEW muon validation
    recoMuonPostProcessorsHLT +
#    NEWrecoMuonPostProcessorsHLT +
    hltMuonPostProcessors
    )

