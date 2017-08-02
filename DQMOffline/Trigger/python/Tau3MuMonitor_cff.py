import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.Tau3MuMonitor_cfi import hltTau3Mumonitoring

HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Monitoring                                             = hltTau3Mumonitoring.clone()
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Monitoring.FolderName                                  = cms.string('HLT/BPH/Tau3Mu/Mu5_Mu1_TkMu1_Tau10')
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Monitoring.taus                                        = cms.InputTag('hltTauPt10MuPts511Mass1p2to2p3Iso', 'Taus') # 3-muon candidates
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Monitoring.GenericTriggerEventPSet.hltPaths            = cms.vstring('HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_v*')

HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_Monitoring                                     = hltTau3Mumonitoring.clone()
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_Monitoring.FolderName                          = cms.string('HLT/BPH/Tau3Mu/Mu5_Mu1_TkMu1_Tau10_Charge1')
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_Monitoring.taus                                = cms.InputTag('hltTauPt10MuPts511Mass1p2to2p3IsoCharge1', 'Taus') # 3-muon candidates, charge = 1
HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_Monitoring.GenericTriggerEventPSet.hltPaths    = cms.vstring('HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_v*')

HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Monitoring                                          = hltTau3Mumonitoring.clone()
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Monitoring.FolderName                               = cms.string('HLT/BPH/Tau3Mu/Mu5_Mu1_TkMu1_IsoTau10')
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Monitoring.taus                                     = cms.InputTag('hltTauPt10MuPts511Mass1p2to2p3Iso', 'SelectedTaus') # 3-muon isolated candidates
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Monitoring.GenericTriggerEventPSet.hltPaths         = cms.vstring('HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_v*')

HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_Monitoring                                  = hltTau3Mumonitoring.clone()
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_Monitoring.FolderName                       = cms.string('HLT/BPH/Tau3Mu/Mu5_Mu1_TkMu1_IsoTau10_Charge1')
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_Monitoring.taus                             = cms.InputTag('hltTauPt10MuPts511Mass1p2to2p3IsoCharge1', 'SelectedTaus') # 3-muon isolated candidates, charge = 1
HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_Monitoring.GenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_v*')

tau3MuMonitorHLT = cms.Sequence(
    HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Monitoring            +
    HLT_Tau3Mu_Mu5_Mu1_TkMu1_Tau10_Charge1_Monitoring    +
    HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Monitoring         +
    HLT_Tau3Mu_Mu5_Mu1_TkMu1_IsoTau10_Charge1_Monitoring
)

