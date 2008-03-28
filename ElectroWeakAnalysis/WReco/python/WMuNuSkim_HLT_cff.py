import FWCore.ParameterSet.Config as cms

# Muons Low L
#include "HLTrigger/Muon/data/PathSingleMu_1032_Iso.cff"
#include "HLTrigger/Muon/data/PathSingleMu_1032_NoIso.cff"
# HWW Skim
WMuNuTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

WMuNuSingleMuFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('WMuNuFilterPath1muIso', 'WMuNuFilterPath1muNoIso'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)

WMuNuHLTPath = cms.Path(WMuNuTrigReport*WMuNuSingleMuFilter)

