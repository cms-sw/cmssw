import FWCore.ParameterSet.Config as cms

HLTTauPostAnalysis = cms.EDAnalyzer("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauOffline/Inclusive/L1",
    "HLT/TauOffline/PFTaus/L1"
    ),
    caloDirs                = cms.vstring(

    ),
    trackDirs               = cms.vstring(

    ),
    pathDirs                = cms.vstring(
    "HLT/TauOffline/PFTaus/DoubleLooseIsoTau",
    "HLT/TauOffline/PFTaus/SingleLooseIsoTau",
    "HLT/TauOffline/PFTaus/EleTau",
    "HLT/TauOffline/PFTaus/MuLooseTau",
    "HLT/TauOffline/PFTaus/MuTightTau",
    ),
    pathSummaryDirs         = cms.vstring(
    "HLT/TauOffline/PFTaus/Summary",
    )
  )
)


HLTTauPostProcess = cms.Sequence(HLTTauPostAnalysis)
