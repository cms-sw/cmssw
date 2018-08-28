import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cff import *
from DQMOffline.Trigger.VBFMETMonitor_cff import *
from DQMOffline.Trigger.HMesonGammaMonitor_cff import *
from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring
from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring
from DQMOffline.Trigger.VBFTauMonitor_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cff import *
from DQMOffline.Trigger.MssmHbbMonitoring_cff import *
from DQMOffline.Trigger.HiggsMonitoring_cfi import hltHIGmonitoring
from DQMOffline.Trigger.BTaggingMonitor_cfi import hltBTVmonitoring

# HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone()
#PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET100_BTag/')
PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/HIG/PFMET100_BTag/')
PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_v")
PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone()
#PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET110_BTag/')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/HIG/PFMET110_BTag/')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_v")
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone()
#PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET110_BTag/')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/HIG/PFMET110_BTag/')
# Selection
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.njets            = cms.uint32(1)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTcut            = cms.double(0)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
# Binning
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_v')
PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET110_PFMHT110_IDTight_v')


# HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone()
#PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET120_BTag/')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/HIG/PFMET120_BTag/')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_v")
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone()
#PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET120_BTag/')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/HIG/PFMET120_BTag/')
# Selection
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.njets            = cms.uint32(1)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTcut            = cms.double(0)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
# Binning
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_v')
PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_v')


# HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone()
#PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET130_BTag/')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/HIG/PFMET130_BTag/')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_v")
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone()
#PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET130_BTag/')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/HIG/PFMET130_BTag/')
# Selection
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.njets            = cms.uint32(1)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTcut            = cms.double(0)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
# Binning
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,130,200,400)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_v')
PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET130_PFMHT130_IDTight_v')


# HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1 MET monitoring
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring = hltMETmonitoring.clone()
#PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET140_BTag/')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring.FolderName = cms.string('HLT/HIG/PFMET140_BTag/')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_v")
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1 b-tag monitoring
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring = hltTOPmonitoring.clone()
#PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET140_BTag/')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.FolderName= cms.string('HLT/HIG/PFMET140_BTag/')
# Selection
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.njets            = cms.uint32(1)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.HTcut            = cms.double(0)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
# Binning
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,140,200,400)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_v')
PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_v')

#######for HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ####
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon = hltHIGmonitoring.clone()
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.nelectrons = cms.uint32(2)
#ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ')
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.FolderName = cms.string('HLT/HIG/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ')
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v*")

##############################DiLepton cross triggers######################################################
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone()
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.nmuons = cms.uint32(1)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.nelectrons = cms.uint32(1)
#mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg')
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg')
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu20_v*","HLT_TkMu20_v*",
	"HLT_IsoMu24_eta2p1_v*",
	"HLT_IsoMu24_v*",
	"HLT_IsoMu27_v*",
	"HLT_IsoMu20_v*",
	"HLT_IsoTkMu24_eta2p1_v*",
	"HLT_IsoTkMu24_v*",
	"HLT_IsoTkMu27_v*",
	"HLT_IsoTkMu20_v*"
)

mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg = hltHIGmonitoring.clone()
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.nmuons = cms.uint32(1)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.nelectrons = cms.uint32(1)
#mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg')
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/HIG/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ/muLeg')
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
        "HLT_Ele27_WPTight_Gsf_v*",
	"HLT_Ele35_WPTight_Gsf_v*"
)

#####HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v#####
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone()
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.nmuons = cms.uint32(1)
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.nelectrons = cms.uint32(1)
#mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg')
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/eleLeg')
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*") #
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu20_v*",
	"HLT_TkMu20_v*",
	"HLT_IsoMu24_eta2p1_v*",
	"HLT_IsoMu24_v*",
	"HLT_IsoMu27_v*",
	"HLT_IsoMu20_v*",
	"HLT_IsoTkMu24_eta2p1_v*",
	"HLT_IsoTkMu24_v*",
	"HLT_IsoTkMu27_v*",
	"HLT_IsoTkMu20_v*"
)

mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg = hltHIGmonitoring.clone()
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.nmuons = cms.uint32(1)
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.nelectrons = cms.uint32(1)
#mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg')
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/HIG/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ/muLeg')
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*")
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
        "HLT_Ele27_WPTight_Gsf_v*",
	"HLT_Ele35_WPTight_Gsf_v*"
)

###############################same flavour trilepton monitor####################################
########TripleMuon########
higgsTrimumon = hltHIGmonitoring.clone()
#higgsTrimumon.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_TripleMu_12_10_5/')
higgsTrimumon.FolderName = cms.string('HLT/HIG/TriLepton/HLT_TripleMu_12_10_5/')
higgsTrimumon.nmuons = cms.uint32(3)
higgsTrimumon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TripleMu_12_10_5_v*") #
higgsTrimumon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*","HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*")

higgsTrimu10_5_5_dz_mon = hltHIGmonitoring.clone()
#higgsTrimu10_5_5_dz_mon.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_TripleM_10_5_5_DZ/')
higgsTrimu10_5_5_dz_mon.FolderName = cms.string('HLT/HIG/TriLepton/HLT_TripleM_10_5_5_DZ/')
higgsTrimu10_5_5_dz_mon.nmuons = cms.uint32(3)
higgsTrimu10_5_5_dz_mon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TripleMu_10_5_5_DZ_v*") #
higgsTrimu10_5_5_dz_mon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*","HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*")

#######TripleElectron####
higgsTrielemon = hltHIGmonitoring.clone()
#higgsTrielemon.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/')
higgsTrielemon.FolderName = cms.string('HLT/HIG/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/')
higgsTrielemon.nelectrons = cms.uint32(3)
higgsTrielemon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v*") #
higgsTrielemon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")

###############################cross flavour trilepton monitor####################################
#########DiMuon+Single Ele Trigger###################
diMu9Ele9CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone()
#diMu9Ele9CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg')
diMu9Ele9CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/muLeg')
diMu9Ele9CaloIdLTrackIdL_muleg.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_muleg.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*")
diMu9Ele9CaloIdLTrackIdL_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
         "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
         "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)

diMu9Ele9CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone()
#diMu9Ele9CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg')
diMu9Ele9CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/eleLeg')
diMu9Ele9CaloIdLTrackIdL_eleleg.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_eleleg.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*")
diMu9Ele9CaloIdLTrackIdL_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*"
)

##Eff of the HLT with DZ w.ref to non-DZ one
diMu9Ele9CaloIdLTrackIdL_dz = hltHIGmonitoring.clone()
#diMu9Ele9CaloIdLTrackIdL_dz.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon')
diMu9Ele9CaloIdLTrackIdL_dz.FolderName = cms.string('HLT/HIG/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL/dzMon')
diMu9Ele9CaloIdLTrackIdL_dz.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_dz.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_dz.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v*")
diMu9Ele9CaloIdLTrackIdL_dz.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*")

#################DiElectron+Single Muon Trigger##################
mu8diEle12CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone()
#mu8diEle12CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg')
mu8diEle12CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/eleLeg')
mu8diEle12CaloIdLTrackIdL_eleleg.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_eleleg.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*")
mu8diEle12CaloIdLTrackIdL_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
       	"HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)

mu8diEle12CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone()
#mu8diEle12CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg')
mu8diEle12CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/muLeg')
mu8diEle12CaloIdLTrackIdL_muleg.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_muleg.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*")
mu8diEle12CaloIdLTrackIdL_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)

##Eff of the HLT with DZ w.ref to non-DZ one
mu8diEle12CaloIdLTrackIdL_dz = hltHIGmonitoring.clone()
#mu8diEle12CaloIdLTrackIdL_dz.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon')
mu8diEle12CaloIdLTrackIdL_dz.FolderName = cms.string('HLT/HIG/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL/dzMon')
mu8diEle12CaloIdLTrackIdL_dz.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_dz.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_dz.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v*")
mu8diEle12CaloIdLTrackIdL_dz.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*")

##VBF triggers##
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1 = hltTOPmonitoring.clone()
#QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v')
# Selection
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.leptJetDeltaRmin = cms.double(0.0)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.njets            = cms.uint32(4)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.jetSelection     = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.HTcut            = cms.double(0)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.nbjets           = cms.uint32(2)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.bjetSelection    = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.btagalgo         = cms.InputTag("pfCombinedMVAV2BJetTags")
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.workingpoint     = cms.double(-0.715) # Loose
# Binning
#QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1_v*')
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 20 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)


QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone()
#QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1_v*')


QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone()
#QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1_v*')


QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1 = QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1.clone()
#QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v')
QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1_v*')


QuadPFJet98_83_71_15_BTagCSV_p013_VBF1 = hltTOPmonitoring.clone()
#QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v')
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v')
# Selection
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.leptJetDeltaRmin = cms.double(0.0)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.njets            = cms.uint32(4)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.jetSelection     = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.HTcut            = cms.double(0)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.nbjets           = cms.uint32(1)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.bjetSelection    = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.btagalgo         = cms.InputTag("pfCombinedMVAV2BJetTags")
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.workingpoint     = cms.double(-0.715) # Loose
# Binning
#QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet98_83_71_15_BTagCSV_p013_VBF2_v*')
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 20 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)

QuadPFJet103_88_75_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone()
#QuadPFJet103_88_75_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v')
QuadPFJet103_88_75_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v')
QuadPFJet103_88_75_15_BTagCSV_p013_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet103_88_75_15_BTagCSV_p013_VBF2_v*')


QuadPFJet105_88_76_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone()
#QuadPFJet105_88_76_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v')
QuadPFJet105_88_76_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v')
QuadPFJet105_88_76_15_BTagCSV_p013_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet105_88_76_15_BTagCSV_p013_VBF2_v*')


QuadPFJet111_90_80_15_BTagCSV_p013_VBF1 = QuadPFJet98_83_71_15_BTagCSV_p013_VBF1.clone()
#QuadPFJet111_90_80_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v')
QuadPFJet111_90_80_15_BTagCSV_p013_VBF1.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v')
QuadPFJet111_90_80_15_BTagCSV_p013_VBF1.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet111_90_80_15_BTagCSV_p013_VBF2_v*')

QuadPFJet98_83_71_15 = hltTOPmonitoring.clone()
#QuadPFJet98_83_71_15.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet98_83_71_15_v')
QuadPFJet98_83_71_15.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet98_83_71_15_v')
# Selection
QuadPFJet98_83_71_15.leptJetDeltaRmin = cms.double(0.0)
QuadPFJet98_83_71_15.njets            = cms.uint32(4)
QuadPFJet98_83_71_15.jetSelection     = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
QuadPFJet98_83_71_15.HTcut            = cms.double(0)
QuadPFJet98_83_71_15.nbjets           = cms.uint32(0)
QuadPFJet98_83_71_15.bjetSelection    = cms.string('pt>15 & abs(eta)<4.7')
QuadPFJet98_83_71_15.btagalgo         = cms.InputTag("pfCombinedMVAV2BJetTags")
QuadPFJet98_83_71_15.workingpoint     = cms.double(-0.715) # Loose
# Binning
#QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
QuadPFJet98_83_71_15.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
QuadPFJet98_83_71_15.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
QuadPFJet98_83_71_15.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
QuadPFJet98_83_71_15.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet98_83_71_15_v*')
QuadPFJet98_83_71_15.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
)
QuadPFJet98_83_71_15.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 20 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)
QuadPFJet98_83_71_15.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32( 1 ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(   1   ),
)

QuadPFJet103_88_75_15 = QuadPFJet98_83_71_15.clone()
#QuadPFJet103_88_75_15.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet103_88_75_15_v')
QuadPFJet103_88_75_15.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet103_88_75_15_v')
QuadPFJet103_88_75_15.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet103_88_75_15_v*')


QuadPFJet105_88_76_15 = QuadPFJet98_83_71_15.clone()
#QuadPFJet105_88_76_15.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet105_88_76_15_v')
QuadPFJet105_88_76_15.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet105_88_76_15_v')
QuadPFJet105_88_76_15.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet105_88_76_15_v*')


QuadPFJet111_90_80_15 = QuadPFJet98_83_71_15.clone()
#QuadPFJet111_90_80_15.FolderName= cms.string('HLT/Higgs/VBFHbb/HLT_QuadPFJet111_90_80_15_v')
QuadPFJet111_90_80_15.FolderName= cms.string('HLT/HIG/VBFHbb/HLT_QuadPFJet111_90_80_15_v')
QuadPFJet111_90_80_15.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_QuadPFJet111_90_80_15_v*')

###############################Higgs Monitor HLT##############################################
higgsMonitorHLT = cms.Sequence(
### THEY WERE IN EXTRA
    higgsinvHLTJetMETmonitoring
  + higgsHLTDiphotonMonitoring
  + higgstautauHLTVBFmonitoring
  + higgsTrielemon
  + higgsTrimumon
  + higgsTrimu10_5_5_dz_mon
  + ele23Ele12CaloIdLTrackIdLIsoVL_dzmon
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg
  + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg
  + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg
  + mu8diEle12CaloIdLTrackIdL_muleg
  + mu8diEle12CaloIdLTrackIdL_eleleg
  + mu8diEle12CaloIdLTrackIdL_dz
  + diMu9Ele9CaloIdLTrackIdL_muleg
  + diMu9Ele9CaloIdLTrackIdL_eleleg
  + diMu9Ele9CaloIdLTrackIdL_dz
  + PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_METmonitoring
  + PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_TOPmonitoring
  + QuadPFJet98_83_71_15_BTagCSV_p013_VBF1
  + QuadPFJet103_88_75_15_BTagCSV_p013_VBF1
  + QuadPFJet105_88_76_15_BTagCSV_p013_VBF1
  + QuadPFJet111_90_80_15_BTagCSV_p013_VBF1
  + QuadPFJet98_83_71_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet103_88_75_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet105_90_76_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet111_90_80_15_DoubleBTagCSV_p013_p08_VBF1
  + QuadPFJet98_83_71_15
  + QuadPFJet103_88_75_15
  + QuadPFJet105_88_76_15
  + QuadPFJet111_90_80_15	
  + mssmHbbBtagTriggerMonitor 
  + mssmHbbMonitorHLT 
  + HMesonGammamonitoring
)


higHLTDQMSourceExtra = cms.Sequence(
)
