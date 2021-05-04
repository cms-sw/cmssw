# Offline DQM for HLT_Mu3er1p5_PFJet100er2p5_PFMETX_PFMHTX_IDTight (X = 70, 80, 90)
# Mateusz Zarucki 2018

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SusyMonitor_cfi import hltSUSYmonitoring

SoftMuHardJetMETSUSYmonitoring = hltSUSYmonitoring.clone()
SoftMuHardJetMETSUSYmonitoring.FolderName = cms.string('HLT/SUSY/SoftMuHardJetMET/')

SoftMuHardJetMETSUSYmonitoring.numGenericTriggerEventPSet.hltInputTag = cms.InputTag("TriggerResults","","HLT")
SoftMuHardJetMETSUSYmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring(
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET70_PFMHT70_IDTight_v*",
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v*",
    "HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v*" 
)

SoftMuHardJetMETSUSYmonitoring.met   = cms.InputTag("pfMetEI")
SoftMuHardJetMETSUSYmonitoring.jets  = cms.InputTag("ak4PFJetsCHS")
SoftMuHardJetMETSUSYmonitoring.muons = cms.InputTag("muons")

SoftMuHardJetMETSUSYmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.5')
SoftMuHardJetMETSUSYmonitoring.leptJetDeltaRmin = cms.double(0.4)
SoftMuHardJetMETSUSYmonitoring.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')

###############
### Muon pt ###
###############
SoftMuHardJetMETSUSYmonitoring_muPt            = SoftMuHardJetMETSUSYmonitoring.clone()
SoftMuHardJetMETSUSYmonitoring_muPt.FolderName = cms.string('HLT/SUSY/SoftMuHardJetMET/Muon')

## Selection ##
SoftMuHardJetMETSUSYmonitoring_muPt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_v*', 'HLT_PFMET130_PFMHT130_IDTight_v*', 'HLT_PFMET140_PFMHT140_IDTight_v*')
# Muon selection
SoftMuHardJetMETSUSYmonitoring_muPt.nmuons       = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_muPt.muoSelection = cms.string('abs(eta)<1.5')
# Jet selection
SoftMuHardJetMETSUSYmonitoring_muPt.njets = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_muPt.jetSelection = cms.string("pt>130 & abs(eta)<2.5")
# MET selection
SoftMuHardJetMETSUSYmonitoring_muPt.enableMETPlot = True
SoftMuHardJetMETSUSYmonitoring_muPt.metSelection = cms.string('pt>150')
SoftMuHardJetMETSUSYmonitoring_muPt.MHTcut       = cms.double(150)

## Binning ##
SoftMuHardJetMETSUSYmonitoring_muPt.histoPSet.muPtBinning = cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)

##############
### Jet pt ###
##############
SoftMuHardJetMETSUSYmonitoring_jetPt = SoftMuHardJetMETSUSYmonitoring.clone()
SoftMuHardJetMETSUSYmonitoring_jetPt.FolderName = cms.string('HLT/SUSY/SoftMuHardJetMET/Jet')

## Selection ##
SoftMuHardJetMETSUSYmonitoring_jetPt.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_IsoMu27_v*")
# Muon selection
SoftMuHardJetMETSUSYmonitoring_jetPt.nmuons       = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_jetPt.muoSelection = cms.string('pt>30 & abs(eta)<1.5')
# Jet selection
SoftMuHardJetMETSUSYmonitoring_jetPt.njets        = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_jetPt.jetSelection = cms.string("abs(eta)<2.5")
# MET selection
SoftMuHardJetMETSUSYmonitoring_jetPt.enableMETPlot = True
SoftMuHardJetMETSUSYmonitoring_jetPt.metSelection  = cms.string('pt>150')
SoftMuHardJetMETSUSYmonitoring_jetPt.MHTcut        = cms.double(150)

# Binning
SoftMuHardJetMETSUSYmonitoring_jetPt.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400) 

##############
### MET pt ###
##############
SoftMuHardJetMETSUSYmonitoring_metPt = SoftMuHardJetMETSUSYmonitoring.clone()
SoftMuHardJetMETSUSYmonitoring_metPt.FolderName = cms.string('HLT/SUSY/SoftMuHardJetMET/MET')

## Selection ##
SoftMuHardJetMETSUSYmonitoring_metPt.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_IsoMu27_v*")
# Muon selection
SoftMuHardJetMETSUSYmonitoring_metPt.nmuons       = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_metPt.muoSelection = cms.string('pt>30 & abs(eta)<1.5')
# Jet selection
SoftMuHardJetMETSUSYmonitoring_metPt.njets        = cms.uint32(1)
SoftMuHardJetMETSUSYmonitoring_metPt.jetSelection = cms.string("pt>130 & abs(eta)<2.5")
# MET selection
SoftMuHardJetMETSUSYmonitoring_metPt.enableMETPlot = True

# Binning
SoftMuHardJetMETSUSYmonitoring_metPt.histoPSet.metPSet = cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300))

susyHLTSoftMuHardJetMETMonitoring = cms.Sequence(
    SoftMuHardJetMETSUSYmonitoring_muPt
  + SoftMuHardJetMETSUSYmonitoring_jetPt
  + SoftMuHardJetMETSUSYmonitoring_metPt
)
