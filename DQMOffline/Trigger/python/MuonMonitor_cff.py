import FWCore.ParameterSet.Config as cms
 
from DQMOffline.Trigger.MuonMonitor_cfi import hltMuonmonitoring


TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/TrkMu12_DoubleTrkMu5NoFiltersNoVtx/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet40_v*","HLT_PFJet60_v*","HLT_PFJet80_v*"]),
    requireValidHLTPaths = False,
)

DoubleMu43NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleMu43NoFiltersNoVtx/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleMu43NoFiltersNoVtx_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)


DoubleMu40NoFiltersNoVtxDisplaced_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleMu40NoFiltersNoVtxDisplaced/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleMu40NoFiltersNoVtxDisplaced_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

#--------------------------------------------------
DoubleL2Mu23NoVtx_2Cha_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL2Mu23NoVtx_2Cha/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu23NoVtx_2Cha_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL2Mu23NoVtx_2Cha_CosmicSeed/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#--------------------------------------------------
DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#--------------------------------------------------
DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#--------------------------------------------------
DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#--------------------------------------------------
DoubleL3Mu16_10NoVtx_DxyMin0p01cm_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL3Mu16_10NoVtx_DxyMin0p01cm/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#--------------------------------------------------
DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm/',
    nmuons = 2,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)



#--------------------------------------------------

Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu43NoFiltersNoVtx_Photon43_CaloIdL/',
    nmuons = 1,
    nelectrons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 43',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 43',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL/',
    nmuons = 1,
    nelectrons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 38',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 38',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)
#####

Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId/',
    nmuons = 1,
    nelectrons = 1,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 38',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Muon/Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 38',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    requireValidHLTPaths = False,
)

exoHLTMuonmonitoring = cms.Sequence(
    TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring 
    + DoubleMu43NoFiltersNoVtx_monitoring
    + DoubleMu40NoFiltersNoVtxDisplaced_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring
    + Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_monitoring
    + Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_MuLeg_monitoring
    + Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_EleLeg_monitoring
    + DoubleL2Mu23NoVtx_2Cha_monitoring
    + DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring
    + DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_monitoring
    + DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_monitoring
    + DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_monitoring
    + DoubleL3Mu16_10NoVtx_DxyMin0p01cm_monitoring
    + DoubleL3dTksMu16_10NoVtx_DxyMin0p01cm_monitoring
)





