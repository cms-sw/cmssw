import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METplusTrackMonitor_cfi import hltMETplusTrackMonitoring

# HLT_MET105_IsoTrk50
MET105_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone(
    FolderName = 'HLT/EXO/MET/MET105_IsoTrk50/',
    hltMetFilter = 'hltMET105::HLT',
    met       = "caloMet",
)
MET105_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET105_IsoTrk50_v*")

# HLT_PFMET105_IsoTrk50
PFMET105_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMET105_IsoTrk50/',
    hltMetFilter = 'hltPFMET105::HLT',
    met       = "caloMet",
)
PFMET105_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET105_IsoTrk50_v*")

# HLT_PFMET110_PFJet100
PFMET110_PFJet100monitoring = hltMETplusTrackMonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMET110_PFJet100/',
    hltMetFilter = 'hltPFMET110::HLT',
    met       = "caloMet",
)
PFMET110_PFJet100monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFJet100_v*")

exoHLTMETplusTrackMonitoring = cms.Sequence(
    MET105_IsoTrk50monitoring
    + PFMET105_IsoTrk50monitoring
    + PFMET110_PFJet100monitoring
)

