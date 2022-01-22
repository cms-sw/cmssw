# Offline DQM for HLT_Mu3er1p5_PFJet100er2p5_PFMETX_PFMHTX_IDTight (X = 70, 80, 90)
# Mateusz Zarucki 2018

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SusyMonitor_cfi import hltSUSYmonitoring

SoftMuHardJetMETSUSYmonitoring = hltSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/SoftMuHardJetMET/',
    met   = "pfMetEI",
    jets  = "ak4PFJetsCHS",
    muons = "muons",
    HTdefinition     = 'pt>30 & abs(eta)<2.5',
    leptJetDeltaRmin = 0.4,
    MHTdefinition    = 'pt>30 & abs(eta)<2.4'
)
SoftMuHardJetMETSUSYmonitoring.numGenericTriggerEventPSet.hltInputTag = cms.InputTag("TriggerResults","","HLT")
SoftMuHardJetMETSUSYmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring(
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET70_PFMHT70_IDTight_v*",
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v*",
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v*" 
)

###############
### Muon pt ###
###############
SoftMuHardJetMETSUSYmonitoring_muPt            = SoftMuHardJetMETSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/SoftMuHardJetMET/Muon',
    # Muon selection
    nmuons       = 1,
    muoSelection = 'abs(eta)<1.5',
    # Jet selection
    njets = 1,
    jetSelection = "pt>130 & abs(eta)<2.5",
    # MET selection
    enableMETPlot = True,
    metSelection = 'pt>150',
    MHTcut       = 150
)
## Selection ##
SoftMuHardJetMETSUSYmonitoring_muPt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_v*', 'HLT_PFMET130_PFMHT130_IDTight_v*', 'HLT_PFMET140_PFMHT140_IDTight_v*')
## Binning ##
SoftMuHardJetMETSUSYmonitoring_muPt.histoPSet.muPtBinning = cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)

##############
### Jet pt ###
##############
SoftMuHardJetMETSUSYmonitoring_jetPt = SoftMuHardJetMETSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/SoftMuHardJetMET/Jet',
    # Muon selection
    nmuons       = 1,
    muoSelection = 'pt>30 & abs(eta)<1.5',
    # Jet selection
    njets        = 1,
    jetSelection = "abs(eta)<2.5",
    # MET selection
    enableMETPlot = True,
    metSelection  = 'pt>150',
    MHTcut        = 150
)
## Selection ##
SoftMuHardJetMETSUSYmonitoring_jetPt.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_IsoMu27_v*")
# Binning
SoftMuHardJetMETSUSYmonitoring_jetPt.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400) 

##############
### MET pt ###
##############
SoftMuHardJetMETSUSYmonitoring_metPt = SoftMuHardJetMETSUSYmonitoring.clone(
    FolderName = 'HLT/SUSY/SoftMuHardJetMET/MET',
    # Muon selection
    nmuons       = 1,
    muoSelection = 'pt>30 & abs(eta)<1.5',
    # Jet selection
    njets        = 1,
    jetSelection = "pt>130 & abs(eta)<2.5",
    # MET selection
    enableMETPlot = True
)
## Selection ##
SoftMuHardJetMETSUSYmonitoring_metPt.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_IsoMu27_v*")
# Binning
SoftMuHardJetMETSUSYmonitoring_metPt.histoPSet.metPSet = cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300))

susyHLTSoftMuHardJetMETMonitoring = cms.Sequence(
    SoftMuHardJetMETSUSYmonitoring_muPt
  + SoftMuHardJetMETSUSYmonitoring_jetPt
  + SoftMuHardJetMETSUSYmonitoring_metPt
)
