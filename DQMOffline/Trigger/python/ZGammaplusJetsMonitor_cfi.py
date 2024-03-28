import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.zgammajetsmonitoring_cfi import zgammajetsmonitoring

hltZJetsmonitoring = zgammajetsmonitoring.clone(
    FolderName = 'HLT/JME/ZGammaPlusJets/',
    PathName = "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_PFJet30_v",
    ModuleName = "hltDiMuon178Mass8Filtered" 
)
