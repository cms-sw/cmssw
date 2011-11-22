import FWCore.ParameterSet.Config as cms

HLTTauValPostAnalysis = cms.EDAnalyzer("HLTTauPostProcessor",
  Harvester = cms.PSet(
    L1Dirs                  = cms.vstring(
    "HLT/TauRelVal/MC_5E32/L1",
    "HLT/TauRelVal/PF_5E32/L1",
    "HLT/TauRelVal/MC_1E33/L1",
    "HLT/TauRelVal/PF_1E33/L1",
    "HLT/TauRelVal/MC_3E33/L1",
    "HLT/TauRelVal/PF_3E33/L1"
    ),
    caloDirs                = cms.vstring(

    ),
    trackDirs               = cms.vstring(

    ),
    pathDirs                = cms.vstring(
    "HLT/TauRelVal/MC_5E32/DoubleTau",
    "HLT/TauRelVal/MC_5E32/SingleTau",
    "HLT/TauRelVal/MC_5E32/EleTau",
    "HLT/TauRelVal/MC_5E32/MuTau",
    "HLT/TauRelVal/PF_5E32/DoubleTau",
    "HLT/TauRelVal/PF_5E32/SingleTau",
    "HLT/TauRelVal/PF_5E32/EleTau",
    "HLT/TauRelVal/PF_5E32/MuTau",
    "HLT/TauRelVal/MC_1E33/DoubleTau",
    "HLT/TauRelVal/MC_1E33/SingleTau",
    "HLT/TauRelVal/MC_1E33/EleTau",
    "HLT/TauRelVal/MC_1E33/MuLooseTau",
	"HLT/TauRelVal/MC_1E33/MuTightTau",
    "HLT/TauRelVal/PF_1E33/DoubleTau",
    "HLT/TauRelVal/PF_1E33/SingleTau",
    "HLT/TauRelVal/PF_1E33/EleTau",
    "HLT/TauRelVal/PF_1E33/MuLooseTau",
	"HLT/TauRelVal/PF_1E33/MuTightTau",
    "HLT/TauRelVal/MC_3E33/DoubleTau",
    "HLT/TauRelVal/MC_3E33/SingleTau",
    "HLT/TauRelVal/MC_3E33/EleTau",
    "HLT/TauRelVal/MC_3E33/MuLooseTau",
    "HLT/TauRelVal/MC_3E33/MuMediumTau",
	"HLT/TauRelVal/MC_3E33/MuTightTau",
    "HLT/TauRelVal/PF_3E33/DoubleTau",
    "HLT/TauRelVal/PF_3E33/SingleTau",
    "HLT/TauRelVal/PF_3E33/EleTau",
    "HLT/TauRelVal/PF_3E33/MuLooseTau",
    "HLT/TauRelVal/PF_3E33/MuMediumTau",
	"HLT/TauRelVal/PF_3E33/MuTightTau",
    ),
    pathSummaryDirs         = cms.vstring(
    'HLT/TauRelVal/MC_5E32/Summary',
    'HLT/TauRelVal/PF_5E32/Summary',
    'HLT/TauRelVal/MC_1E33/Summary',
    'HLT/TauRelVal/PF_1E33/Summary',
    'HLT/TauRelVal/MC_3E33/Summary',
    'HLT/TauRelVal/PF_3E33/Summary'
    )
  )
)

HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
