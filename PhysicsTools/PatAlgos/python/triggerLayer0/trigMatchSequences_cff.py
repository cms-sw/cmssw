import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.triggerLayer0.patTrigMatcher_cfi import *

# demo sequences to have a few trigger match examples in the
# default PAT configuration
patTrigMatch_withoutBTau = cms.Sequence(
    patTrigMatchCandHLT1ElectronStartup +
    patTrigMatchHLT1PhotonRelaxed +
    patTrigMatchHLT1ElectronRelaxed +
    patTrigMatchHLT1MuonNonIso +
    patTrigMatchHLT2jet +
    patTrigMatchHLT1MET65
)

patTrigMatch = cms.Sequence(
    patTrigMatch_withoutBTau +
    patTrigMatchHLT1Tau
)


## patTuple ##

patTrigMatch_patTuple_withoutBTau = cms.Sequence(
    patTrigMatchHLT_IsoMu11 +
    patTrigMatchHLT_Mu11 +
    patTrigMatchHLT_DoubleIsoMu3 +
    patTrigMatchHLT_DoubleMu3 +
    patTrigMatchHLT_IsoEle15_LW_L1I +
    patTrigMatchHLT_Ele15_LW_L1R +
    patTrigMatchHLT_DoubleIsoEle10_LW_L1I +
    patTrigMatchHLT_DoubleEle5_SW_L1R
)

patTrigMatch_patTuple = cms.Sequence(
    patTrigMatch_patTuple_withoutBTau +
    patTrigMatchHLT_LooseIsoTau_MET30_L1MET +
    patTrigMatchHLT_DoubleIsoTau_Trk3
)