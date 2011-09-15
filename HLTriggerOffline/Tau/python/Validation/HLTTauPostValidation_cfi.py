import FWCore.ParameterSet.Config as cms

HLTTauValPostAnalysis = cms.EDAnalyzer("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauRelVal/MC_6E31/L1",
    "HLT/TauRelVal/PF_6E31/L1"
    ),
    caloDirs                = cms.vstring(
    "HLT/TauRelVal/MC_6E31/L2",
    "HLT/TauRelVal/PF_6E31/L2"

    ),
    trackDirs               = cms.vstring(
    "HLT/TauRelVal/MC_6E31/L25",
    "HLT/TauRelVal/MC_6E31/L3",
    "HLT/TauRelVal/PF_6E31/L25",
    "HLT/TauRelVal/PF_6E31/L3"

    ),
    pathDirs                = cms.vstring(
    "HLT/TauRelVal/MC_6E31/DoubleTau",
    "HLT/TauRelVal/MC_6E31/SingleTau",
    "HLT/TauRelVal/PF_6E31/DoubleTau",
    "HLT/TauRelVal/PF_6E31/SingleTau"

    ),
    pathSummaryDirs         = cms.vstring(
    'HLT/TauRelVal/MC_6E31/Summary',
    'HLT/TauRelVal/PF_6E31/Summary'
    )
  )
)


HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
