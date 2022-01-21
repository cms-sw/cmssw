import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METplusTrackMonitor_cfi import hltMETplusTrackMonitoring

# HLT_MET105_IsoTrk50
MET105_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone(
    FolderName = 'HLT/MET/MET105_IsoTrk50/',
    hltMetFilter = 'hltMET105::HLT',
    hltMetCleanFilter = 'hltMETClean65::HLT'
)
MET105_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET105_IsoTrk50_v*")
# HLT_MET120_IsoTrk50
MET120_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone(
    FolderName = 'HLT/MET/MET120_IsoTrk50/',
    hltMetFilter = 'hltMET120::HLT',
    hltMetCleanFilter = 'hltMETClean65::HLT'
)
MET120_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET120_IsoTrk50_v*")
exoHLTMETplusTrackMonitoring = cms.Sequence(
    MET105_IsoTrk50monitoring
    + MET120_IsoTrk50monitoring
)

