
import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HiggsMonitoring_cfi import hltHIGmonitoring

##############################DiLepton cross triggers######################################################
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref = hltHIGmonitoring.clone()
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref.nmuons = cms.uint32(1)
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref.nelectrons = cms.uint32(1)
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_muref/')
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v*") 
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
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

mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref = hltHIGmonitoring.clone()
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref.nmuons = cms.uint32(1)
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref.nelectrons = cms.uint32(1)
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_eleref/')
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v*")
mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
        "HLT_Ele27_WPTight_Gsf_v*",
	"HLT_Ele35_WPTight_Gsf_v*"
)

#####HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v#####
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref = hltHIGmonitoring.clone()
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref.nmuons = cms.uint32(1)
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref.nelectrons = cms.uint32(1)
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_muref/')
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*") # 
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
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

mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref = hltHIGmonitoring.clone()
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref.nmuons = cms.uint32(1)
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref.nelectrons = cms.uint32(1)
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_eleref/')
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*")
mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
        "HLT_Ele27_WPTight_Gsf_v*",
	"HLT_Ele35_WPTight_Gsf_v*"
)

###############################same flavour trilepton monitor####################################
########TripleMuon########
higgsTrimumon = hltHIGmonitoring.clone()
higgsTrimumon.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_TripleMu_12_10_5/')
higgsTrimumon.nmuons = cms.uint32(3)
higgsTrimumon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TripleMu_12_10_5_v*") # 
higgsTrimumon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*","HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*")

#######TripleElectron####
higgsTrielemon = hltHIGmonitoring.clone()
higgsTrielemon.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL/')
higgsTrielemon.nelectrons = cms.uint32(3)
higgsTrielemon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v*") # 
higgsTrielemon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")

###############################cross flavour trilepton monitor####################################
#########DiMuon+Single Ele Trigger###################
diMu9Ele9CaloIdLTrackIdL_muegref = hltHIGmonitoring.clone()
diMu9Ele9CaloIdLTrackIdL_muegref.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL_muegRef/')
diMu9Ele9CaloIdLTrackIdL_muegref.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_muegref.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_muegref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
diMu9Ele9CaloIdLTrackIdL_muegref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v*", 
	"HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*",
	"HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v*", 
        "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*"
)

diMu9Ele9CaloIdLTrackIdL_dimuref = hltHIGmonitoring.clone()
diMu9Ele9CaloIdLTrackIdL_dimuref.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL_dimuRef/')
diMu9Ele9CaloIdLTrackIdL_dimuref.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_dimuref.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_dimuref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
diMu9Ele9CaloIdLTrackIdL_dimuref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*"
)

#################DiElectron+Single Muon Trigger##################
mu8diEle12CaloIdLTrackIdL_muegref = hltHIGmonitoring.clone()
mu8diEle12CaloIdLTrackIdL_muegref.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL_muegRef/')
mu8diEle12CaloIdLTrackIdL_muegref.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_muegref.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_muegref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
mu8diEle12CaloIdLTrackIdL_muegref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v*", 
	"HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*",
	"HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v*", 
        "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*"
)

mu8diEle12CaloIdLTrackIdL_dieleref = hltHIGmonitoring.clone()
mu8diEle12CaloIdLTrackIdL_dieleref.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL_dieleRef/')
mu8diEle12CaloIdLTrackIdL_dieleref.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_dieleref.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_dieleref.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
mu8diEle12CaloIdLTrackIdL_dieleref.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)


###############################Higgs Monitor HLT##############################################
higgsMonitorHLT = cms.Sequence(
 higgsTrielemon
 + higgsTrimumon
 + mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_muref
 + mu23TrkIsoVVLEle8CaloIdLTrackIdLIsoVL_eleref
 + mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_muref
 + mu8TrkIsoVVLEle23CaloIdLTrackIdLIsoVL_eleref
 + mu8diEle12CaloIdLTrackIdL_muegref
 + mu8diEle12CaloIdLTrackIdL_dieleref
 + diMu9Ele9CaloIdLTrackIdL_dimuref
 + diMu9Ele9CaloIdLTrackIdL_muegref
)
