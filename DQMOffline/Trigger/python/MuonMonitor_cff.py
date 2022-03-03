import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MuonMonitor_cfi import hltMuonmonitoring


TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/TrkMu12_DoubleTrkMu5NoFiltersNoVtx/'
)
TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*")
TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet40_v*","HLT_PFJet60_v*","HLT_PFJet80_v*") 

TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/TrkMu16_DoubleTrkMu6NoFiltersNoVtx/'
)
TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v*")
TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*") 

TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/TrkMu17_DoubleTrkMu8NoFiltersNoVtx/'
)
TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v*")
TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*")

DoubleMu43NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleMu43NoFiltersNoVtx/',
    nmuons = 2
)
DoubleMu43NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu43NoFiltersNoVtx_v*")
DoubleMu43NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


DoubleMu48NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleMu48NoFiltersNoVtx/',
    nmuons = 2
)
DoubleMu48NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu48NoFiltersNoVtx_v*")
DoubleMu48NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


DoubleMu33NoFiltersNoVtxDisplaced_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleMu33NoFiltersNoVtxDisplaced/',
    nmuons = 2
)
DoubleMu33NoFiltersNoVtxDisplaced_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu33NoFiltersNoVtxDisplaced_v*")
DoubleMu33NoFiltersNoVtxDisplaced_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

DoubleMu40NoFiltersNoVtxDisplaced_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleMu40NoFiltersNoVtxDisplaced/',
    nmuons = 2
)
DoubleMu40NoFiltersNoVtxDisplaced_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu40NoFiltersNoVtxDisplaced_v*")
DoubleMu40NoFiltersNoVtxDisplaced_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

#--------------------------------------------------
DoubleL2Mu23NoVtx_2Cha_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleL2Mu23NoVtx_2Cha/',
    nmuons = 2
)
DoubleL2Mu23NoVtx_2Cha_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu23NoVtx_2Cha_v*")
DoubleL2Mu23NoVtx_2Cha_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/DoubleL2Mu23NoVtx_2Cha_CosmicSeed/',
    nmuons = 2
)
DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v*")
DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
#--------------------------------------------------

Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL/',
    nmuons = 1,
    nelectrons = 1
)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 43'
)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 43'
)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL/',
    nmuons = 1,
    nelectrons = 1
)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 48'
)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 48'
)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL/',
    nmuons = 1,
    nelectrons = 1
)
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*")
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 38'
)
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*")
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 38'
)
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v*")
Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL/',
    nmuons = 1,
    nelectrons = 1
)
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_MuLeg/',
    nmuons = 1,
    nelectrons = 1,
    eleSelection = 'pt > 43'
)
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone(
    FolderName = 'HLT/EXO/Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_EleLeg/',
    nmuons = 1,
    nelectrons = 1,
    muonSelection = 'pt > 43'
)
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

exoHLTMuonmonitoring = cms.Sequence(
    TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring 
    + TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring
    + TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring
    + DoubleMu43NoFiltersNoVtx_monitoring
    + DoubleMu48NoFiltersNoVtx_monitoring
    + DoubleMu33NoFiltersNoVtxDisplaced_monitoring
    + DoubleMu40NoFiltersNoVtxDisplaced_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_monitoring
    + Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_MuLeg_monitoring
    + Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_MuLeg_monitoring
    + Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_EleLeg_monitoring
    + Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_EleLeg_monitoring
    + DoubleL2Mu23NoVtx_2Cha_monitoring
    + DoubleL2Mu23NoVtx_2Cha_CosmicSeed_monitoring
)





