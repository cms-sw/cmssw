import FWCore.ParameterSet.Config as cms

HLTTauValPostAnalysis = cms.EDFilter("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauRelVal/MC_8E29/L1",
    "HLT/TauRelVal/MC_1E31/L1",
    "HLT/TauRelVal/PF_8E29/L1",
    "HLT/TauRelVal/PF_1E31/L1"
    ),
    caloDirs                = cms.vstring(
    "HLT/TauRelVal/MC_8E29/L2",
    "HLT/TauRelVal/MC_1E31/L2",
    "HLT/TauRelVal/PF_8E29/L2",
    "HLT/TauRelVal/PF_1E31/L2"

    ),
    trackDirs               = cms.vstring(
    "HLT/TauRelVal/MC_1E31/L25",
    "HLT/TauRelVal/MC_1E31/L3",
    "HLT/TauRelVal/PF_1E31/L25",
    "HLT/TauRelVal/PF_1E31/L3"

    ),
    pathDirs                = cms.vstring(
    "HLT/TauRelVal/MC_8E29/DoubleTau",
    "HLT/TauRelVal/MC_1E31/DoubleTau",
    "HLT/TauRelVal/MC_8E29/SingleTau",
    "HLT/TauRelVal/MC_1E31/SingleTau",
    "HLT/TauRelVal/PF_8E29/DoubleTau",
    "HLT/TauRelVal/PF_1E31/DoubleTau",
    "HLT/TauRelVal/PF_8E29/SingleTau",
    "HLT/TauRelVal/PF_1E31/SingleTau"

    ),
    pathSummaryDirs         = cms.vstring(
    'HLT/TauRelVal/MC_8E29/Summary',
    'HLT/TauRelVal/MC_1E31/Summary'
    'HLT/TauRelVal/PF_8E29/Summary',
    'HLT/TauRelVal/PF_1E31/Summary'
    )
  )
)


HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
