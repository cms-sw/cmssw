import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.RazorMonitor_cff import *
from DQMOffline.Trigger.VBFSUSYMonitor_cff import *
from DQMOffline.Trigger.LepHTMonitor_cff import *
from DQMOffline.Trigger.susyHLTEleCaloJets_cff import *
from DQMOffline.Trigger.SoftMuHardJetMETSUSYMonitor_cff import *
from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

# muon
double_soft_muon_muonpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/Muon/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    enableMETPlot = True,
    metSelection     ='pt>150',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut  = 50,
    invMassLowercut  = 10
)
# Binning
double_soft_muon_muonpt.histoPSet.muPtBinning      =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
double_soft_muon_muonpt.histoPSet.muPtBinning2D    =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
# Triggers
double_soft_muon_muonpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_muonpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_v*')

# met
double_soft_muon_metpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/MET/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.4',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut  = 50,
    invMassLowercut  = 10,
    enableMETPlot    = True
)
# Binning
double_soft_muon_metpt.histoPSet.metPSet   =cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# inv Mass
double_soft_muon_mll = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/Mll/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>10 & abs(eta)<2.4',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut            = 150,
    enableMETPlot = True,
    metSelection      = 'pt>150'
)
# Binning
double_soft_muon_mll.histoPSet.invMassVariableBinning      =cms.vdouble(8,12,15,20,25,30,35,40,45,47,50,60)

# Triggers
double_soft_muon_mll.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_mll.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Dimuon12_Upsilon_eta1p5_v*')

# mht
double_soft_muon_mhtpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/MHT/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.0',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    enableMETPlot = True,
    metSelection     = 'pt>150',
    invMassUppercut       = 50,
    invMassLowercut       = 10
)
# Binning
double_soft_muon_mhtpt.histoPSet.MHTVariableBinning      =cms.vdouble(50,60,70,80,90,100,110,120,130,150,200,300)

# Triggers
double_soft_muon_mhtpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_mhtpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# backup1, met
double_soft_muon_backup_70_metpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/backup70/MET/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.4',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut       = 50,
    invMassLowercut       = 10,
    enableMETPlot = True
)
# Binning
double_soft_muon_backup_70_metpt.histoPSet.metPSet   =cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_backup_70_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v*')
double_soft_muon_backup_70_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# backup1, mht
double_soft_muon_backup_70_mhtpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/backup70/MHT/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.0',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    enableMETPlot = True,
    metSelection     = 'pt>150',
    invMassUppercut       = 50,
    invMassLowercut       = 10
)
# Binning
double_soft_muon_backup_70_mhtpt.histoPSet.MHTVariableBinning      =cms.vdouble(50,60,70,80,90,100,110,120,130,150,200,300)

# Triggers
double_soft_muon_backup_70_mhtpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v*')
double_soft_muon_backup_70_mhtpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# backup2, met
double_soft_muon_backup_90_metpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/backup90/MET/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.4',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut       = 50,
    invMassLowercut       = 10,
    enableMETPlot = True
)
# Binning
double_soft_muon_backup_90_metpt.histoPSet.metPSet   =cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_backup_90_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v*')
double_soft_muon_backup_90_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# backup2, mht
double_soft_muon_backup_90_mhtpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/backup90/MHT/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.0',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    enableMETPlot = True,
    metSelection     = 'pt>150',
    invMassUppercut       = 50,
    invMassLowercut       = 10
)
# Binning
double_soft_muon_backup_90_mhtpt.histoPSet.MHTVariableBinning      =cms.vdouble(50,60,70,80,90,100,110,120,130,150,200,300)
# Triggers
double_soft_muon_backup_90_mhtpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v*')
double_soft_muon_backup_90_mhtpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

# triple muon
triple_muon_mupt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/TripleMu/Muon',
    # Selections
    nmuons           = 3,
    muoSelection     = 'isGlobalMuon()',
    invMassUppercut       = 50,
    invMassLowercut       = 10,
    invMassCutInAllMuPairs= True
)
# Triggers
triple_muon_mupt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_TripleMu_5_3_3_Mass3p8to60_DZ_v*')
triple_muon_mupt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Trimuon5_3p5_2_Upsilon_Muon_v*')

# triplemu dca
triple_muon_dca_mupt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/TripleMu/DCA/Muon',
    # Selections
    nmuons           = 3,
    muoSelection     = 'isGlobalMuon()',
    invMassUppercut       = 50,
    invMassLowercut       = 10,
    invMassCutInAllMuPairs =True
)
# Triggers
triple_muon_dca_mupt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_TripleMu_5_3_3_Mass3p8to60_DCA_v*')
triple_muon_dca_mupt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Trimuon5_3p5_2_Upsilon_Muon_v*')

# MuonEG
susyMuEGMonitoring = hltTOPmonitoring.clone(
    FolderName = 'HLT/SUSY/MuonEG/',
    nmuons = 1,
    nphotons = 1,
    nelectrons = 0,
    njets = 0,
    enablePhotonPlot =  True,
    muoSelection = 'pt>26 & abs(eta)<2.1 & isPFMuon & isGlobalMuon & isTrackerMuon & numberOfMatches>1  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0  & globalTrack.hitPattern.numberOfValidMuonHits>0 & globalTrack.normalizedChi2<10 & (pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.) )/pt<0.15',
    phoSelection = '(pt > 30 && abs(eta)<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295 && neutralHadronIso < 5.931+0.0163*pt+0.000014*pt*pt && photonIso < 6.641+0.0034*pt) || (pt > 30 && abs(eta)>1.4442 && hadTowOverEm<0.0481 && full5x5_sigmaIetaIeta()<0.03013 && chargedHadronIso<1.011 && neutralHadronIso < 1.715+0.0163*pt+0.000014*pt*pt && photonIso < 3.863+0.0034*pt)'
)
susyMuEGMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_Photon30_IsoCaloId*')
susyMuEGMonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('')

# muon dca
double_soft_muon_dca_muonpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/DCA/Muon/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    enableMETPlot = True,
    metSelection     = 'pt>150',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut  = 50,
    invMassLowercut  = 10
)
# Binning
double_soft_muon_dca_muonpt.histoPSet.muPtBinning      =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
double_soft_muon_dca_muonpt.histoPSet.muPtBinning2D    =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
# Triggers
double_soft_muon_dca_muonpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v*')
double_soft_muon_dca_muonpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_v*')

# met
double_soft_muon_dca_metpt = hltTOPmonitoring.clone(
    FolderName   = 'HLT/SUSY/SOS/DCA/MET/',
    # Selections
    nmuons           = 2,
    HTdefinition     = 'pt>30 & abs(eta)<2.4',
    HTcut            = 60,
    muoSelection     = 'pt>18 & abs(eta)<2.4',
    MHTdefinition    = 'pt>30 & abs(eta)<2.4',
    MHTcut           = 150,
    invMassUppercut       = 50,
    invMassLowercut       = 10,
    enableMETPlot = True
)
# Binning
double_soft_muon_dca_metpt.histoPSet.metPSet   =cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_dca_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v*')
double_soft_muon_dca_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*')

susyMonitorHLT = cms.Sequence(
    susyHLTRazorMonitoring
  + susyHLTVBFMonitoring
  + LepHTMonitor
  + susyHLTEleCaloJets
  + double_soft_muon_muonpt
  + double_soft_muon_metpt
  + double_soft_muon_mhtpt
  + double_soft_muon_mll
  + double_soft_muon_backup_70_metpt
  + double_soft_muon_backup_70_mhtpt
  + double_soft_muon_backup_90_metpt
  + double_soft_muon_backup_90_mhtpt
  + triple_muon_mupt
  + triple_muon_dca_mupt
  + susyMuEGMonitoring 
  + double_soft_muon_dca_muonpt
  + double_soft_muon_dca_metpt
  + susyHLTSoftMuHardJetMETMonitoring
)

susHLTDQMSourceExtra = cms.Sequence(
)
