import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "PoolSource", 
    fileNames = cms.untracked.vstring(
    "/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_341_v2/0006/A2767553-B9ED-DE11-9381-00248C0BE014.root",
    "/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_341_v2/0004/D86E6614-93ED-DE11-89F1-001A92971B32.root",
    "/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_341_v2/0004/BEB59F3E-99ED-DE11-A552-003048678F6C.root",
    )
)
