
import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HiggsMonitoring_cfi import hltHIGmonitoring

#######for HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ####
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon = hltHIGmonitoring.clone()
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.nelectrons = cms.uint32(2)
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ')
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*") 
ele23Ele12CaloIdLTrackIdLIsoVL_dzmon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v*")

##############################DiLepton cross triggers######################################################
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone()
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.nmuons = cms.uint32(1)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.nelectrons = cms.uint32(1)
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_eleLeg')
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
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_muref/')
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*")
mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
        "HLT_Ele27_WPTight_Gsf_v*",
	"HLT_Ele35_WPTight_Gsf_v*"
)

#####HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v#####
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg = hltHIGmonitoring.clone()
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.nmuons = cms.uint32(1)
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.nelectrons = cms.uint32(1)
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_eleLeg/')
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
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.FolderName = cms.string('HLT/Higgs/DiLepton/HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_muLeg/')
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*")
mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
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
diMu9Ele9CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone()
diMu9Ele9CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL_muLeg/')
diMu9Ele9CaloIdLTrackIdL_muleg.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_muleg.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
diMu9Ele9CaloIdLTrackIdL_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
         "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
         "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*" 
)

diMu9Ele9CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone()
diMu9Ele9CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_DiMu9_Ele9_CaloIdL_TrackIdL_eleLeg/')
diMu9Ele9CaloIdLTrackIdL_eleleg.nelectrons = cms.uint32(1)
diMu9Ele9CaloIdLTrackIdL_eleleg.nmuons = cms.uint32(2)
diMu9Ele9CaloIdLTrackIdL_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
diMu9Ele9CaloIdLTrackIdL_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*",
	"HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*"
)

#################DiElectron+Single Muon Trigger##################
mu8diEle12CaloIdLTrackIdL_eleleg = hltHIGmonitoring.clone()
mu8diEle12CaloIdLTrackIdL_eleleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL_eleLeg/')
mu8diEle12CaloIdLTrackIdL_eleleg.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_eleleg.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_eleleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
mu8diEle12CaloIdLTrackIdL_eleleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*",
       	"HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)

mu8diEle12CaloIdLTrackIdL_muleg = hltHIGmonitoring.clone()
mu8diEle12CaloIdLTrackIdL_muleg.FolderName = cms.string('HLT/Higgs/TriLepton/HLT_Mu8_DiEle12_CaloIdL_TrackIdL_muLeg/')
mu8diEle12CaloIdLTrackIdL_muleg.nelectrons = cms.uint32(2)
mu8diEle12CaloIdLTrackIdL_muleg.nmuons = cms.uint32(1)
mu8diEle12CaloIdLTrackIdL_muleg.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*") # HLT_ZeroBias_v*
mu8diEle12CaloIdLTrackIdL_muleg.denGenericTriggerEventPSet.hltPaths = cms.vstring(
	"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"
)


###############################Higgs Monitor HLT##############################################
higgsMonitorHLT = cms.Sequence(
 higgsTrielemon
 + higgsTrimumon
 + ele23Ele12CaloIdLTrackIdLIsoVL_dzmon
 + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_eleleg
 + mu23TrkIsoVVLEle12CaloIdLTrackIdLIsoVLDZ_muleg
 + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_eleleg
 + mu12TrkIsoVVLEle23CaloIdLTrackIdLIsoVLDZ_muleg
 + mu8diEle12CaloIdLTrackIdL_muleg
 + mu8diEle12CaloIdLTrackIdL_eleleg
 + diMu9Ele9CaloIdLTrackIdL_muleg
 + diMu9Ele9CaloIdLTrackIdL_eleleg
)
