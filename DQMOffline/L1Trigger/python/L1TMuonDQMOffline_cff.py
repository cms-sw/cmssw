import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TMuonDQMOffline_cfi import *

# modifications for the pp reference run
muonEfficiencyThresholds_HI = [5, 7, 12]
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tMuonDQMOffline,
    gmtPtCuts = cms.untracked.vint32(muonEfficiencyThresholds_HI),
    tagPtCut = cms.untracked.double(14.),
    triggerNames = cms.untracked.vstring(
        "HLT_HIL3Mu12_v*",
    )
)

