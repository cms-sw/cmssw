import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METplusTrackMonitor_cfi import hltMETplusTrackMonitoring

# HLT_MET105_IsoTrk50
MET105_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone()
MET105_IsoTrk50monitoring.FolderName = cms.string('HLT/MET/MET105_IsoTrk50/')
MET105_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET105_IsoTrk50_v")
MET105_IsoTrk50monitoring.hltMetSelection = cms.string("pt > 105")
MET105_IsoTrk50monitoring.hltMetCleanSelection = cms.string("pt > 65")

# HLT_MET120_IsoTrk50
MET120_IsoTrk50monitoring = hltMETplusTrackMonitoring.clone()
MET120_IsoTrk50monitoring.FolderName = cms.string('HLT/MET/MET120_IsoTrk50/')
MET120_IsoTrk50monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET120_IsoTrk50_v")
MET105_IsoTrk50monitoring.hltMetSelection = cms.string("pt > 120")
MET105_IsoTrk50monitoring.hltMetCleanSelection = cms.string("pt > 65")

exoHLTMETplusTrackMonitoring = cms.Sequence(
    MET105_IsoTrk50monitoring
    + MET120_IsoTrk50monitoring
)

