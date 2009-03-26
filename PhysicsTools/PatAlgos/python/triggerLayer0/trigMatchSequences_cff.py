import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.triggerLayer0.patTrigMatcher_cfi import *

# demo sequences to have a few trigger match examples in the
# default PAT configuration
patTrigMatch = cms.Sequence(
    patTrigMatchCandHLT1ElectronStartup +
    patTrigMatchHLT1PhotonRelaxed +
    patTrigMatchHLT1ElectronRelaxed +
    patTrigMatchHLT1MuonNonIso +
    patTrigMatchHLT2jet +
    patTrigMatchHLT1MET65 +
    patTrigMatchHLT1Tau
)
