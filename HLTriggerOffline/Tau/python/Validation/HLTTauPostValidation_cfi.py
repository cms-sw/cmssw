import FWCore.ParameterSet.Config as cms
HLTTauValPostAnalysis = cms.EDFilter("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauRelVal/MC/L1"
    ),
    caloDirs                = cms.vstring(
    "HLT/TauRelVal/MC/L2"
    ),
    trackDirs               = cms.vstring(
    "HLT/TauRelVal/MC/L25",
    "HLT/TauRelVal/MC/L3"
    ),
    pathDirs                = cms.vstring(
    "HLT/TauRelVal/MC/DoubleTau",
    "HLT/TauRelVal/MC/SingleTau"
    ),
    pathSummaryDirs         = cms.vstring(
    'HLT/TauRelVal/MC/Summary'
    )
  )
)


HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
