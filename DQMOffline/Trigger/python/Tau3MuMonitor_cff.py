import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.Tau3MuMonitor_cfi import hltTau3Mumonitoring

HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Monitoring  = hltTau3Mumonitoring.clone(
    FolderName   = 'HLT/BPH/Tau3Mu/Mu7_Mu1_TkMu1_Tau15',
    taus         = 'hltTauPt15MuPts711Mass1p3to2p1Iso:Taus',# 3-muon candidates
    GenericTriggerEventPSet = dict(hltPaths = ['HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v*'])
)

HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_Monitoring  = hltTau3Mumonitoring.clone(
    FolderName = 'HLT/BPH/Tau3Mu/Mu7_Mu1_TkMu1_Tau15_Charge1',
    taus       = 'hltTauPt15MuPts711Mass1p3to2p1IsoCharge1:Taus', # 3-muon candidates, charge = 1
    GenericTriggerEventPSet = dict(hltPaths    = ['HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v*'])
)

HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Monitoring  = hltTau3Mumonitoring.clone(
    FolderName   = 'HLT/BPH/Tau3Mu/Mu7_Mu1_TkMu1_IsoTau15',
    taus        = 'hltTauPt15MuPts711Mass1p3to2p1Iso:SelectedTaus', # 3-muon isolated candidates
    GenericTriggerEventPSet = dict(hltPaths = ['HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v*'])
)

HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_Monitoring  = hltTau3Mumonitoring.clone(
    FolderName = 'HLT/BPH/Tau3Mu/Mu7_Mu1_TkMu1_IsoTau15_Charge1',
    taus = 'hltTauPt15MuPts711Mass1p3to2p1IsoCharge1:SelectedTaus', # 3-muon isolated candidates, charge = 1
    GenericTriggerEventPSet = dict(hltPaths = ['HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v*'])
)

tau3MuMonitorHLT = cms.Sequence(
    HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Monitoring            +
    HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_Monitoring    +
    HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Monitoring         +
    HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_Monitoring
)

