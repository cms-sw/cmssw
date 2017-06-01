import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BPHMonitor_cfi import hltBPHmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
Dimuon0_Jpsi_tnp = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack2/')
Dimuon0_Jpsi_tnp.tnp = cms.int32(1)
Dimuon0_Jpsi_tnp.nofset = cms.int32(1)
Dimuon0_Jpsi_tnp.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v1")
Dimuon0_Jpsi_tnp.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_Jpsi_tnp.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track2_Jpsi_v4")
Dimuon0_Jpsi_tnp.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Jpsi_tnp.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_Upsilon_tnp = hltBPHmonitoring.clone()
Dimuon0_Upsilon_tnp.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_NO_OS_denTrack2/')
Dimuon0_Upsilon_tnp.tnp = cms.int32(1)
Dimuon0_Upsilon_tnp.nofset = cms.int32(1)
Dimuon0_Upsilon_tnp.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_NoOS_v1")
Dimuon0_Upsilon_tnp.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5_SQ")
Dimuon0_Upsilon_tnp.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track2_Upsilon_v4")
Dimuon0_Upsilon_tnp.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Upsilon_tnp.muoSelection = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Upsilon_tnp.muoSelection_ref = cms.string("pt>7.5 abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
##
Dimuon0_Jpsi_tnp1 = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp1.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack7/')
Dimuon0_Jpsi_tnp1.tnp = cms.int32(1)
Dimuon0_Jpsi_tnp1.nofset = cms.int32(1)
Dimuon0_Jpsi_tnp1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v1")
Dimuon0_Jpsi_tnp1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_Jpsi_tnp1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track7_Jpsi_v4")
Dimuon0_Jpsi_tnp1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Jpsi_tnp1.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp1.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_Upsilon_tnp1 = hltBPHmonitoring.clone()
Dimuon0_Upsilon_tnp1.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_NO_OS_denTrack7/')
Dimuon0_Upsilon_tnp1.tnp = cms.int32(1)
Dimuon0_Upsilon_tnp1.nofset = cms.int32(1)
Dimuon0_Upsilon_tnp1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_NoOS_v1")
Dimuon0_Upsilon_tnp1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5_SQ")
Dimuon0_Upsilon_tnp1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track7_Upsilon_v4")
Dimuon0_Upsilon_tnp1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Upsilon_tnp1.muoSelection = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Upsilon_tnp1.muoSelection_ref = cms.string("pt>7.5 abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
##
##
Dimuon0_Jpsi_tnp2 = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp2.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack3p5/')
Dimuon0_Jpsi_tnp2.tnp = cms.int32(1)
Dimuon0_Jpsi_tnp2.nofset = cms.int32(1)
Dimuon0_Jpsi_tnp2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v1")
Dimuon0_Jpsi_tnp2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_Jpsi_tnp2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track3p5_Jpsi_v4")
Dimuon0_Jpsi_tnp2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Jpsi_tnp2.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp2.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_Upsilon_tnp2 = hltBPHmonitoring.clone()
Dimuon0_Upsilon_tnp2.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_NO_OS_denTrack3p5/')
Dimuon0_Upsilon_tnp2.tnp = cms.int32(1)
Dimuon0_Upsilon_tnp2.nofset = cms.int32(1)
Dimuon0_Upsilon_tnp2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_NoOS_v1")
Dimuon0_Upsilon_tnp2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5_SQ")
Dimuon0_Upsilon_tnp2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track3p5_Upsilon_v4")
Dimuon0_Upsilon_tnp2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("SingleMu5_SQ OR SingleMu7_SQ")
Dimuon0_Upsilon_tnp2.muoSelection = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Upsilon_tnp2.muoSelection_ref = cms.string("pt>7.5 abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
##

##

Dimuon0_Jpsi_OS = hltBPHmonitoring.clone()
Dimuon0_Jpsi_OS.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_OS/')
Dimuon0_Jpsi_OS.tnp = cms.int32(0)
Dimuon0_Jpsi_OS.nofset = cms.int32(2)
Dimuon0_Jpsi_OS.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v1")
Dimuon0_Jpsi_OS.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ_OS")
Dimuon0_Jpsi_OS.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v1")
Dimuon0_Jpsi_OS.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_Jpsi_OS.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_OS.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###

Dimuon0_er = hltBPHmonitoring.clone()
Dimuon0_er.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_er/')
Dimuon0_er.tnp = cms.int32(0)
Dimuon0_er.nofset = cms.int32(3)
Dimuon0_er.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v1")
Dimuon0_er.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0er1p5_SQ_OS")
Dimuon0_er.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v1")
Dimuon0_er.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_er.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_er.muoSelection = cms.string("abs(eta)<1.5 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###
Dimuon0_Upsilon_er = hltBPHmonitoring.clone()
Dimuon0_Upsilon_er.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_er/')
Dimuon0_Upsilon_er.tnp = cms.int32(0)
Dimuon0_Upsilon_er.nofset = cms.int32(3)## if 3 ad pt cut in .cc file
Dimuon0_Upsilon_er.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0_v1")
Dimuon0_Upsilon_er.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5er2p0_SQ_OS")
Dimuon0_Upsilon_er.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5_v1")
Dimuon0_Upsilon_er.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5_SQ_OS")
Dimuon0_Upsilon_er.muoSelection_ref = cms.string("abs(eta)<2. &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Upsilon_er.muoSelection = cms.string("abs(eta)<2. & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###
Dimuon0_dRcut = hltBPHmonitoring.clone()
Dimuon0_dRcut.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_dR/')
Dimuon0_dRcut.tnp = cms.int32(0)
Dimuon0_dRcut.nofset = cms.int32(4)
Dimuon0_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5R_v1")
Dimuon0_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0er1p5_SQ_OS_dR_1p4")
Dimuon0_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v1")
Dimuon0_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0er1p5_SQ_OS")
Dimuon0_dRcut.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_dRcut.muoSelection = cms.string("abs(eta)<1.5 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###
Dimuon0_dRcut_low = hltBPHmonitoring.clone()
Dimuon0_dRcut_low.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_dR_low/')
Dimuon0_dRcut_low.tnp = cms.int32(0)
Dimuon0_dRcut_low.nofset = cms.int32(4)
Dimuon0_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4R_v1")
Dimuon0_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2")
Dimuon0_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v1")
Dimuon0_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS")
Dimuon0_dRcut_low.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_dRcut_low.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###
###mass cut
Dimuon0_masscut1 = hltBPHmonitoring.clone()
Dimuon0_masscut1.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_masscut1/')
Dimuon0_masscut1.tnp = cms.int32(0)
Dimuon0_masscut1.nofset = cms.int32(5)
Dimuon0_masscut1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v1")
Dimuon0_masscut1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon0_masscut1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0_v1")
Dimuon0_masscut1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5er2p0_SQ_OS")
Dimuon0_masscut1.muoSelection_ref = cms.string("abs(eta)<2. &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_masscut1.muoSelection = cms.string("abs(eta)<2. & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_masscut2 = hltBPHmonitoring.clone()
Dimuon0_masscut2.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_masscut2/')
Dimuon0_masscut2.tnp = cms.int32(0)
Dimuon0_masscut2.nofset = cms.int32(5)
Dimuon0_masscut2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_5M_v1")
Dimuon0_masscut2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu5_SQ_OS_Mass_7to18")
Dimuon0_masscut2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_5_v1")
Dimuon0_masscut2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu5_SQ_OS")
Dimuon0_masscut2.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_masscut2.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###triple muon

Dimuon0_tripleMu1 = hltBPHmonitoring.clone()
Dimuon0_tripleMu1.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu1/')
Dimuon0_tripleMu1.tnp = cms.int32(0)
Dimuon0_tripleMu1.nofset = cms.int32(6)
Dimuon0_tripleMu1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Trimuon2_Upsilon5_Muon_NoL1Mass_v1")
Dimuon0_tripleMu1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu5_3p5_2")
Dimuon0_tripleMu1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v1")
Dimuon0_tripleMu1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu5_3p5_2")
Dimuon0_tripleMu1.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu1.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_tripleMu2 = hltBPHmonitoring.clone()
Dimuon0_tripleMu2.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu2/')
Dimuon0_tripleMu2.tnp = cms.int32(0)
Dimuon0_tripleMu2.nofset = cms.int32(6)
Dimuon0_tripleMu2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v1")
Dimuon0_tripleMu2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu5_3p5_2")
Dimuon0_tripleMu2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_Muon")
Dimuon0_tripleMu2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu0_Qopen")
Dimuon0_tripleMu2.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu2.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")


Dimuon0_tripleMu3 = hltBPHmonitoring.clone()
Dimuon0_tripleMu3.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu3/')
Dimuon0_tripleMu3.tnp = cms.int32(0)
Dimuon0_tripleMu3.nofset = cms.int32(6)
Dimuon0_tripleMu3.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v1")
Dimuon0_tripleMu3.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu_5SQ_3SQ_0OQ")
Dimuon0_tripleMu3.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_TM530_v1")
Dimuon0_tripleMu3.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("TripleMu_5SQ_3SQ_0OQ")
Dimuon0_tripleMu3.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu3.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###photon 
Dimuon0_photon1 = hltBPHmonitoring.clone()
Dimuon0_photon1.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_photon1/')
Dimuon0_photon1.tnp = cms.int32(0)
Dimuon0_photon1.nofset = cms.int32(7)
Dimuon0_photon1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_Photon23_v1")
Dimuon0_photon1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_EG12")
Dimuon0_photon1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v1")
Dimuon0_photon1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_EG12")
Dimuon0_photon1.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_photon1.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_photon2 = hltBPHmonitoring.clone()
Dimuon0_photon2.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_photon2/')
Dimuon0_photon2.tnp = cms.int32(0)
Dimuon0_photon2.nofset = cms.int32(7)
Dimuon0_photon2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v1")
Dimuon0_photon2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_EG12")
Dimuon0_photon2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4_v1")
Dimuon0_photon2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS")
Dimuon0_photon2.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_photon2.muoSelection = cms.string("abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###
###L3 TnP
Dimuon0_L3TnP_Jpsi = hltBPHmonitoring.clone()
Dimuon0_L3TnP_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_L3TnP_Jpsi/')
Dimuon0_L3TnP_Jpsi.tnp = cms.int32(1)
Dimuon0_L3TnP_Jpsi.nofset = cms.int32(1)
Dimuon0_L3TnP_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Jpsi_v5")
Dimuon0_L3TnP_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_L3TnP_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Jpsi_v5")
Dimuon0_L3TnP_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_L3TnP_Jpsi.muoSelection_ref = cms.string("pt>7.5 &  abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_L3TnP_Jpsi.muoSelection = cms.string("pt> 7.5 & abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_L3TnP_Upsilon = hltBPHmonitoring.clone()
Dimuon0_L3TnP_Upsilon.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_L3TnP_Upsilon/')
Dimuon0_L3TnP_Upsilon.tnp = cms.int32(1)
Dimuon0_L3TnP_Upsilon.nofset = cms.int32(1)
Dimuon0_L3TnP_Upsilon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Upsilon_v5")
Dimuon0_L3TnP_Upsilon.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_L3TnP_Upsilon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Upsilon_v5")
Dimuon0_L3TnP_Upsilon.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu0_SQ")
Dimuon0_L3TnP_Upsilon.muoSelection_ref = cms.string("pt>7.5 &  abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_L3TnP_Upsilon.muoSelection = cms.string("pt> 7.5 & abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###
###HLT OS 

Dimuon0_HLT_OS = hltBPHmonitoring.clone()
Dimuon0_HLT_OS.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_HLT_OS/')
Dimuon0_HLT_OS.tnp = cms.int32(0)
Dimuon0_HLT_OS.nofset = cms.int32(2)
Dimuon0_HLT_OS.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v1")
Dimuon0_HLT_OS.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_HLT_OS.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v1")
Dimuon0_HLT_OS.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_HLT_OS.muoSelection_ref = cms.string(" abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_HLT_OS.muoSelection = cms.string(" abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###
###Loose vertex Jpsi
Dimuon0_looseVtx_Jpsi = hltBPHmonitoring.clone()
Dimuon0_looseVtx_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_looseVtx_Jpsi/')
Dimuon0_looseVtx_Jpsi.tnp = cms.int32(0)
Dimuon0_looseVtx_Jpsi.nofset = cms.int32(8)
Dimuon0_looseVtx_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v1")
Dimuon0_looseVtx_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_looseVtx_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v1")
Dimuon0_looseVtx_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_looseVtx_Jpsi.muoSelection_ref = cms.string(" abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_looseVtx_Jpsi.muoSelection = cms.string(" abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
####Loose vtx Upsilon
Dimuon0_looseVtx_Upsilon = hltBPHmonitoring.clone()
Dimuon0_looseVtx_Upsilon.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_looseVtx_Upsilon/')
Dimuon0_looseVtx_Upsilon.tnp = cms.int32(0)
Dimuon0_looseVtx_Upsilon.nofset = cms.int32(8)
Dimuon0_looseVtx_Upsilon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v1")
Dimuon0_looseVtx_Upsilon.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon0_looseVtx_Upsilon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_NoVertexing_v1")
Dimuon0_looseVtx_Upsilon.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon0_looseVtx_Upsilon.muoSelection_ref = cms.string(" abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_looseVtx_Upsilon.muoSelection = cms.string(" abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###tight vtx
Dimuon0_tightVtx_Jpsi = hltBPHmonitoring.clone()
Dimuon0_tightVtx_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_tightVtx_Jpsi/')
Dimuon0_tightVtx_Jpsi.tnp = cms.int32(0)
Dimuon0_tightVtx_Jpsi.nofset = cms.int32(8)
Dimuon0_tightVtx_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_displaced_v1")
Dimuon0_tightVtx_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_tightVtx_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v1")
Dimuon0_tightVtx_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_tightVtx_Jpsi.muoSelection_ref = cms.string(" abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tightVtx_Jpsi.muoSelection = cms.string(" abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###additional track
Dimuon0_addTrack_Jpsi = hltBPHmonitoring.clone()
Dimuon0_addTrack_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_addTrack_Jpsi/')
Dimuon0_addTrack_Jpsi.tnp = cms.int32(0)
Dimuon0_addTrack_Jpsi.nofset = cms.int32(9)
Dimuon0_addTrack_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_displaced_v7")
Dimuon0_addTrack_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrack_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_displaced_v1")
Dimuon0_addTrack_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("DoubleMu4_SQ_OS_dR_Max1p2 OR DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrack_Jpsi.muoSelection_ref = cms.string(" abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_addTrack_Jpsi.muoSelection = cms.string(" abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

###

bphHLTmonitoring = cms.Sequence(
    Dimuon0_Upsilon_tnp
    + Dimuon0_Upsilon_tnp1
    + Dimuon0_Upsilon_tnp2
    + Dimuon0_Jpsi_OS
    + Dimuon0_er
    + Dimuon0_Upsilon_er
    + Dimuon0_dRcut
    + Dimuon0_dRcut_low
    + Dimuon0_masscut1
    + Dimuon0_masscut2
    + Dimuon0_tripleMu1
    + Dimuon0_tripleMu2
    + Dimuon0_tripleMu3
    + Dimuon0_photon1
    + Dimuon0_photon2
    + Dimuon0_L3TnP_Jpsi
    + Dimuon0_L3TnP_Upsilon
    + Dimuon0_HLT_OS
    + Dimuon0_looseVtx_Jpsi
    + Dimuon0_looseVtx_Upsilon
    + Dimuon0_tightVtx_Jpsi
    + Dimuon0_addTrack_Jpsi
)



bphMonitorHLT = cms.Sequence(
    bphHLTmonitoring
)

