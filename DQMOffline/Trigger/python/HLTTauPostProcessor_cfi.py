import FWCore.ParameterSet.Config as cms

HLTTauPostAnalysis = cms.EDAnalyzer("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauOffline/Inclusive/L1",
    "HLT/TauOffline/PFTaus/L1",
    "HLT/TauOffline/Photons/L1",
    "HLT/TauOffline/HPD/L1"
    ),
    caloDirs                = cms.vstring(
    "HLT/TauOffline/Inclusive/L2",
    "HLT/TauOffline/PFTaus/L2",
    "HLT/TauOffline/Photons/L2",
    "HLT/TauOffline/HPD/L2"
    ),
    trackDirs               = cms.vstring(

    ),
    pathDirs                = cms.vstring(
    "HLT/TauOffline/PFTaus/DoubleLooseIsoTau",
    "HLT/TauOffline/Photons/DoubleLooseIsoTau",
    "HLT/TauOffline/HPD/DoubleLooseIsoTau",
    "HLT/TauOffline/PFTaus/SingleLooseIsoTau",
    "HLT/TauOffline/Photons/SingleLooseIsoTau",
    "HLT/TauOffline/HPD/SingleLooseIsoTau"
    ),
    pathSummaryDirs         = cms.vstring(
    "HLT/TauOffline/PFTaus/Summary",
    "HLT/TauOffline/Photons/Summary",
    "HLT/TauOffline/HPD/Summary"
    )
  )
)


HLTTauPostProcess = cms.Sequence(HLTTauPostAnalysis)
