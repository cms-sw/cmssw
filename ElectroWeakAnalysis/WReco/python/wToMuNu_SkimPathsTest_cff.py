import FWCore.ParameterSet.Config as cms

# HLT setup
from HLTrigger.Configuration.common.HLTSetup_cff import *
# Muons Low L
from HLTrigger.Muon.PathSingleMu_1032_Iso_cff import *
from HLTrigger.Muon.PathSingleMu_1032_NoIso_cff import *
WMuNuFilterPath1muIso = cms.Path(singleMuIso)
WMuNuFilterPath1muNoIso = cms.Path(singleMuNoIso)

