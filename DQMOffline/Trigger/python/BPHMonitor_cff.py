import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BPHMonitor_cfi import hltBPHmonitoring

# Tau3Mu
from DQMOffline.Trigger.Tau3MuMonitor_cff import *

Dimuon0_Jpsi_tnp = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack2/')
Dimuon0_Jpsi_tnp.tnp = cms.bool(True)
Dimuon0_Jpsi_tnp.enum = cms.int32(1)
Dimuon0_Jpsi_tnp.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*")
#Dimuon0_Jpsi_tnp.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_Jpsi_tnp.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track2_Jpsi_v*")
#Dimuon0_Jpsi_tnp.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_SingleMu5_SQ OR L1_SingleMu7_SQ")
Dimuon0_Jpsi_tnp.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon25_Jpsi_tnp = hltBPHmonitoring.clone()
Dimuon25_Jpsi_tnp.FolderName = cms.string('HLT/BPH/DiMu25_Jpsi_noCorr/')
Dimuon25_Jpsi_tnp.tnp = cms.bool(True)
Dimuon25_Jpsi_tnp.enum = cms.int32(1)
Dimuon25_Jpsi_tnp.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon25_Jpsi_noCorrL1_v*")
#Dimuon25_Jpsi_tnp.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu8_SQ")
Dimuon25_Jpsi_tnp.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon25_Jpsi_tnp.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

##
Dimuon0_Jpsi_tnp1 = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp1.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack7/')
Dimuon0_Jpsi_tnp1.tnp = cms.bool(True)
Dimuon0_Jpsi_tnp1.enum = cms.int32(1)
Dimuon0_Jpsi_tnp1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*")
#Dimuon0_Jpsi_tnp1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_Jpsi_tnp1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track7_Jpsi_v*")
#Dimuon0_Jpsi_tnp1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_SingleMu5_SQ OR L1_SingleMu7_SQ")
Dimuon0_Jpsi_tnp1.muoSelection_ref = cms.string("pt>7.5 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp1.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

##
Dimuon0_Jpsi_tnp2 = hltBPHmonitoring.clone()
Dimuon0_Jpsi_tnp2.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_NO_OS_denTrack3p5/')
Dimuon0_Jpsi_tnp2.tnp = cms.bool(True)
Dimuon0_Jpsi_tnp2.enum = cms.int32(1)
Dimuon0_Jpsi_tnp2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*")
#Dimuon0_Jpsi_tnp2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_Jpsi_tnp2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_Track3p5_Jpsi_v*")
#Dimuon0_Jpsi_tnp2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_SingleMu5_SQ OR L1_SingleMu7_SQ")
Dimuon0_Jpsi_tnp2.muoSelection_ref = cms.string("pt>7.5 && abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_tnp2.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")


##

Dimuon0_Jpsi_OS = hltBPHmonitoring.clone()
Dimuon0_Jpsi_OS.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_OS/')
Dimuon0_Jpsi_OS.tnp = cms.bool(False)
Dimuon0_Jpsi_OS.enum = cms.int32(2)
Dimuon0_Jpsi_OS.Jpsi = cms.int32(1)
Dimuon0_Jpsi_OS.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v*")
#Dimuon0_Jpsi_OS.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ_OS")
Dimuon0_Jpsi_OS.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*")
#Dimuon0_Jpsi_OS.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_Jpsi_OS.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Jpsi_OS.DMSelection_ref = cms.string("abs(Eta)<2.4")

###

Dimuon0_er = hltBPHmonitoring.clone()
Dimuon0_er.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_er/')
Dimuon0_er.tnp = cms.bool(False)
Dimuon0_er.enum = cms.int32(3)
Dimuon0_er.Jpsi = cms.int32(1)
Dimuon0_er.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#Dimuon0_er.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
Dimuon0_er.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v*")
#Dimuon0_er.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ_OS")
Dimuon0_er.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_er.DMSelection_ref = cms.string("abs(Eta)<1.5")

###
Dimuon0_Upsilon_er = hltBPHmonitoring.clone()
Dimuon0_Upsilon_er.FolderName = cms.string('HLT/BPH/DiMu0_Upsilon_L1_er/')
Dimuon0_Upsilon_er.tnp = cms.bool(False)
Dimuon0_Upsilon_er.enum = cms.int32(3)##
Dimuon0_Upsilon_er.Upsilon = cms.int32(1)
Dimuon0_Upsilon_er.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0_v*")
#Dimuon0_Upsilon_er.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS")
Dimuon0_Upsilon_er.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5_v*")
#Dimuon0_Upsilon_er.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5_SQ_OS")
Dimuon0_Upsilon_er.muoSelection_ref = cms.string("abs(eta)<2. &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_Upsilon_er.DMSelection_ref = cms.string("abs(Eta)<2. ")

###L1 dR cut

###
DMu4_3_Bs_dRcut = hltBPHmonitoring.clone()
DMu4_3_Bs_dRcut.FolderName = cms.string('HLT/BPH/DMu4_3_Bs_L1_dR/')
DMu4_3_Bs_dRcut.tnp = cms.bool(False)
DMu4_3_Bs_dRcut.enum = cms.int32(4)
DMu4_3_Bs_dRcut.ptCut = cms.int32(1)
DMu4_3_Bs_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_3_Bs_v*")
#DMu4_3_Bs_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
DMu4_3_Bs_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#DMu4_3_Bs_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
DMu4_3_Bs_dRcut.muoSelection_ref = cms.string("pt>4.5 && abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_3_Bs_dRcut.muoSelection = cms.string("pt>3.5 && abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_3_Bs_dRcut.DMSelection_ref = cms.string("abs(Eta)<1.5 && Pt>5.5 && M<5.9 && M>4.6")
DMu4_3_Bs_dRcut.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32( 30 ),
  xmin  = cms.double(0.4),
  xmax  = cms.double(1.9),
)
DMu4_3_Bs_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32(9),
  xmin  = cms.double(  4.4),
  xmax  = cms.double(6.2),
)


DMu4_3_Jpsi_dRcut = hltBPHmonitoring.clone()
DMu4_3_Jpsi_dRcut.FolderName = cms.string('HLT/BPH/DMu4_3_Jpsi_L1_dR/')
DMu4_3_Jpsi_dRcut.tnp = cms.bool(False)
DMu4_3_Jpsi_dRcut.enum = cms.int32(4)
DMu4_3_Jpsi_dRcut.Jpsi = cms.int32(1)
DMu4_3_Jpsi_dRcut.ptCut = cms.int32(1)
DMu4_3_Jpsi_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_3_Jpsi_v*")
#DMu4_3_Jpsi_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
DMu4_3_Jpsi_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#DMu4_3_Jpsi_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
DMu4_3_Jpsi_dRcut.muoSelection_ref = cms.string("pt>4.5 &&  abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_3_Jpsi_dRcut.muoSelection = cms.string("pt>3.5 && abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_3_Jpsi_dRcut.DMSelection_ref = cms.string("abs(Eta)<1.5")
DMu4_3_Jpsi_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(2.8),
  xmax  = cms.double(3.3),
)


Dimuon14_Phi_dRcut = hltBPHmonitoring.clone()
Dimuon14_Phi_dRcut.FolderName = cms.string('HLT/BPH/DiMu14_Phi_L1_dR/')
Dimuon14_Phi_dRcut.tnp = cms.bool(False)
Dimuon14_Phi_dRcut.enum = cms.int32(4)
Dimuon14_Phi_dRcut.seagull = cms.int32(1)
Dimuon14_Phi_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon14_Phi_Barrel_Seagulls_v*")
#Dimuon14_Phi_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
Dimuon14_Phi_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#Dimuon14_Phi_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
Dimuon14_Phi_dRcut.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon14_Phi_dRcut.DMSelection_ref = cms.string("M>0.95 & M<1.1 & Pt>15 & abs(Eta)<1.2")
Dimuon14_Phi_dRcut.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32( 35 ),
  xmin  = cms.double(0),
  xmax  = cms.double(0.7),
)
Dimuon14_Phi_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(0.95),
  xmax  = cms.double(1.1),
)


DMu4_LowMassNonResonantTrk_Displaced_dRcut = hltBPHmonitoring.clone()
DMu4_LowMassNonResonantTrk_Displaced_dRcut.FolderName = cms.string('HLT/BPH/DMu4_LowMassNonResonantTrk_Displaced_L1_dR/')
DMu4_LowMassNonResonantTrk_Displaced_dRcut.tnp = cms.bool(False)
DMu4_LowMassNonResonantTrk_Displaced_dRcut.enum = cms.int32(4)
DMu4_LowMassNonResonantTrk_Displaced_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v*")
#DMu4_LowMassNonResonantTrk_Displaced_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
DMu4_LowMassNonResonantTrk_Displaced_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#DMu4_LowMassNonResonantTrk_Displaced_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
DMu4_LowMassNonResonantTrk_Displaced_dRcut.muoSelection_ref = cms.string("pt>5 & abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_LowMassNonResonantTrk_Displaced_dRcut.DMSelection_ref = cms.string("((M>1.1 & M<2.8) | (M>4.1 & M<4.7)) & abs(Eta)<1.5")
DMu4_LowMassNonResonantTrk_Displaced_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(1.),
  xmax  = cms.double(3.),
)

DMu4_LowMassNonResonantTrk_Displaced_dRcut_low = hltBPHmonitoring.clone()
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.FolderName = cms.string('HLT/BPH/DMu4_LowMassNonResonantTrk_Displaced_L1_dR_low/')
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.tnp = cms.bool(False)
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.enum = cms.int32(4)
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v*")
#DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2")
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v*")
#DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.muoSelection_ref = cms.string("pt>5 & abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.DMSelection_ref = cms.string("((M>1.1 & M<2.8) | (M>4.1 & M<4.7)) & abs(Eta)<2.4")
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.max_dR = cms.double(1.2)
DMu4_LowMassNonResonantTrk_Displaced_dRcut_low.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(1.),
  xmax  = cms.double(3.),
)


DMu4_JpsiTrk_Displaced_dRcut = hltBPHmonitoring.clone()
DMu4_JpsiTrk_Displaced_dRcut.FolderName = cms.string('HLT/BPH/DMu4_JpsiTrk_Displaced_L1_dR/')
DMu4_JpsiTrk_Displaced_dRcut.tnp = cms.bool(False)
DMu4_JpsiTrk_Displaced_dRcut.enum = cms.int32(4)
DMu4_JpsiTrk_Displaced_dRcut.Jpsi = cms.int32(1)
DMu4_JpsiTrk_Displaced_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_Displaced_v*")
#DMu4_JpsiTrk_Displaced_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
DMu4_JpsiTrk_Displaced_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#DMu4_JpsiTrk_Displaced_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
DMu4_JpsiTrk_Displaced_dRcut.muoSelection_ref = cms.string("pt>5 & abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_JpsiTrk_Displaced_dRcut.DMSelection_ref = cms.string("abs(Eta)<1.5")
DMu4_JpsiTrk_Displaced_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(2.8),
  xmax  = cms.double(3.3),
)


DMu4_JpsiTrk_Displaced_dRcut_low = hltBPHmonitoring.clone()
DMu4_JpsiTrk_Displaced_dRcut_low.FolderName = cms.string('HLT/BPH/DMu4_JpsiTrk_Displaced_L1_dR_low/')
DMu4_JpsiTrk_Displaced_dRcut_low.tnp = cms.bool(False)
DMu4_JpsiTrk_Displaced_dRcut_low.enum = cms.int32(4)
DMu4_JpsiTrk_Displaced_dRcut_low.Jpsi = cms.int32(1)
DMu4_JpsiTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_Displaced_v*")
#DMu4_JpsiTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2")
DMu4_JpsiTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v*")
#DMu4_JpsiTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
DMu4_JpsiTrk_Displaced_dRcut_low.muoSelection_ref = cms.string("pt>5 & abs(eta)<2.4  &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_JpsiTrk_Displaced_dRcut_low.DMSelection_ref = cms.string("abs(Eta)<2.4")
DMu4_JpsiTrk_Displaced_dRcut_low.max_dR = cms.double(1.2)
DMu4_JpsiTrk_Displaced_dRcut_low.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(2.8),
  xmax  = cms.double(3.3),
)

DMu4_PsiPrimeTrk_Displaced_dRcut = hltBPHmonitoring.clone()
DMu4_PsiPrimeTrk_Displaced_dRcut.FolderName = cms.string('HLT/BPH/DMu4_PsiPrimeTrk_Displaced_L1_dR/')
DMu4_PsiPrimeTrk_Displaced_dRcut.tnp = cms.bool(False)
DMu4_PsiPrimeTrk_Displaced_dRcut.enum = cms.int32(4)
DMu4_PsiPrimeTrk_Displaced_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_PsiPrimeTrk_Displaced_v*")
#DMu4_PsiPrimeTrk_Displaced_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
DMu4_PsiPrimeTrk_Displaced_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#DMu4_PsiPrimeTrk_Displaced_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
DMu4_PsiPrimeTrk_Displaced_dRcut.muoSelection_ref = cms.string("pt>5 & abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_PsiPrimeTrk_Displaced_dRcut.DMSelection_ref = cms.string("M>3.4 & M<3.95 & abs(Eta)<1.5")
DMu4_PsiPrimeTrk_Displaced_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 12 ),
  xmin  = cms.double(3.3),
  xmax  = cms.double(3.9),
)



DMu4_PsiPrimeTrk_Displaced_dRcut_low = hltBPHmonitoring.clone()
DMu4_PsiPrimeTrk_Displaced_dRcut_low.FolderName = cms.string('HLT/BPH/DMu4_PsiPrimeTrk_Displaced_L1_dR_low/')
DMu4_PsiPrimeTrk_Displaced_dRcut_low.tnp = cms.bool(False)
DMu4_PsiPrimeTrk_Displaced_dRcut_low.enum = cms.int32(4)
DMu4_PsiPrimeTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_PsiPrimeTrk_Displaced_v*")
#DMu4_PsiPrimeTrk_Displaced_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2")
DMu4_PsiPrimeTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v*")
#DMu4_PsiPrimeTrk_Displaced_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
DMu4_PsiPrimeTrk_Displaced_dRcut_low.muoSelection_ref = cms.string("pt>5 & abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DMu4_PsiPrimeTrk_Displaced_dRcut_low.DMSelection_ref = cms.string("M>3.4 & M<3.95 & abs(Eta)<2.4")
DMu4_PsiPrimeTrk_Displaced_dRcut_low.max_dR = cms.double(1.2)
DMu4_PsiPrimeTrk_Displaced_dRcut_low.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 12 ),
  xmin  = cms.double(3.3),
  xmax  = cms.double(3.9),
)


Dimuon25_Jpsi_dRcut = hltBPHmonitoring.clone()
Dimuon25_Jpsi_dRcut.FolderName = cms.string('HLT/BPH/DiMu25_Jpsi_L1_dR/')
Dimuon25_Jpsi_dRcut.tnp = cms.bool(False)
Dimuon25_Jpsi_dRcut.enum = cms.int32(4)
Dimuon25_Jpsi_dRcut.Jpsi = cms.int32(1)
Dimuon25_Jpsi_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon25_Jpsi_v*")
#Dimuon25_Jpsi_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
Dimuon25_Jpsi_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#Dimuon25_Jpsi_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
Dimuon25_Jpsi_dRcut.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon25_Jpsi_dRcut.DMSelection_ref = cms.string("Pt>26 & abs(Eta)<1.5")
Dimuon25_Jpsi_dRcut.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32( 20 ),
  xmin  = cms.double(0.),
  xmax  = cms.double(1.),
)
Dimuon25_Jpsi_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(2.8),
  xmax  = cms.double(3.3),
)


Dimuon18_PsiPrime_dRcut = hltBPHmonitoring.clone()
Dimuon18_PsiPrime_dRcut.FolderName = cms.string('HLT/BPH/DiMu18_PsiPrime_L1_dR/')
Dimuon18_PsiPrime_dRcut.tnp = cms.bool(False)
Dimuon18_PsiPrime_dRcut.enum = cms.int32(4)
Dimuon18_PsiPrime_dRcut.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon18_PsiPrime_v*")
#Dimuon18_PsiPrime_dRcut.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS_dR_1p4")
Dimuon18_PsiPrime_dRcut.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_0er1p5_v*")
#Dimuon18_PsiPrime_dRcut.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0er1p5_SQ_OS")
Dimuon18_PsiPrime_dRcut.muoSelection_ref = cms.string("abs(eta)<1.5 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon18_PsiPrime_dRcut.DMSelection_ref = cms.string("M>3.4 & M<3.95 & Pt>19 & abs(Eta)<1.5")
Dimuon18_PsiPrime_dRcut.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 12 ),
  xmin  = cms.double(3.3),
  xmax  = cms.double(3.9),
)

Dimuon25_Jpsi_dRcut_low = hltBPHmonitoring.clone()
Dimuon25_Jpsi_dRcut_low.FolderName = cms.string('HLT/BPH/DiMu25_Jpsi_L1_dR_low/')
Dimuon25_Jpsi_dRcut_low.tnp = cms.bool(False)
Dimuon25_Jpsi_dRcut_low.enum = cms.int32(4)
Dimuon25_Jpsi_dRcut_low.Jpsi = cms.int32(1)
Dimuon25_Jpsi_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon25_Jpsi_v*")
#Dimuon25_Jpsi_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2")
Dimuon25_Jpsi_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v*")
#Dimuon25_Jpsi_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
Dimuon25_Jpsi_dRcut_low.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon25_Jpsi_dRcut_low.DMSelection_ref = cms.string("Pt>26 & abs(Eta)<2.4")
Dimuon25_Jpsi_dRcut_low.max_dR = cms.double(1.2)
Dimuon25_Jpsi_dRcut_low.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32( 18 ),
  xmin  = cms.double(0.),
  xmax  = cms.double(0.9),
)
Dimuon25_Jpsi_dRcut_low.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(2.8),
  xmax  = cms.double(3.3),
)


Dimuon18_PsiPrime_dRcut_low = hltBPHmonitoring.clone()
Dimuon18_PsiPrime_dRcut_low.FolderName = cms.string('HLT/BPH/DiMu18_PsiPrime_L1_dR_low/')
Dimuon18_PsiPrime_dRcut_low.tnp = cms.bool(False)
Dimuon18_PsiPrime_dRcut_low.enum = cms.int32(4)
Dimuon18_PsiPrime_dRcut_low.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon18_PsiPrime_v*")
#Dimuon18_PsiPrime_dRcut_low.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2")
Dimuon18_PsiPrime_dRcut_low.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_4_v*")
#Dimuon18_PsiPrime_dRcut_low.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
Dimuon18_PsiPrime_dRcut_low.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon18_PsiPrime_dRcut_low.DMSelection_ref = cms.string("M>3.4 & M<3.95 & Pt>19 & abs(Eta)<2.4")
Dimuon18_PsiPrime_dRcut_low.max_dR = cms.double(1.2)
Dimuon18_PsiPrime_dRcut_low.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32( 10 ),
  xmin  = cms.double(3.4),
  xmax  = cms.double(4.0),
)

###
###mass cut
Dimuon20_masscut1 = hltBPHmonitoring.clone()
Dimuon20_masscut1.FolderName = cms.string('HLT/BPH/DiMu20_Upsilon_L1_masscut1/')
Dimuon20_masscut1.tnp = cms.bool(False)
Dimuon20_masscut1.Upsilon = cms.int32(1)
Dimuon20_masscut1.enum = cms.int32(5)
Dimuon20_masscut1.seagull = cms.int32(1)
Dimuon20_masscut1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon10_Upsilon_Barrel_Seagulls_v*")
#Dimuon20_masscut1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon20_masscut1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0_v*")
#Dimuon20_masscut1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS")
Dimuon20_masscut1.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.0 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon20_masscut1.DMSelection_ref = cms.string("M<18 & M>7 & Pt>11 & abs(Eta)<1.2")



Dimuon12_masscut2 = hltBPHmonitoring.clone()
Dimuon12_masscut2.FolderName = cms.string('HLT/BPH/DiMu12_Upsilon_L1_masscut2/')
Dimuon12_masscut2.tnp = cms.bool(False)
Dimuon12_masscut2.enum = cms.int32(5)
Dimuon12_masscut2.Upsilon = cms.int32(1)
Dimuon12_masscut2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon12_Upsilon_y1p4_v*")
#Dimuon12_masscut2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon12_masscut2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0_v*")
#Dimuon12_masscut2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS")
Dimuon12_masscut2.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.0 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon12_masscut2.DMSelection_ref = cms.string("Pt>13 & abs(y)<1.5")


Trimuon2_masscut4 = hltBPHmonitoring.clone()
Trimuon2_masscut4.FolderName = cms.string('HLT/BPH/TripleMu2_Upsilon_L1_masscut4')
Trimuon2_masscut4.tnp = cms.bool(False)
Trimuon2_masscut4.enum = cms.int32(5)
Trimuon2_masscut4.Upsilon = cms.int32(1)
Trimuon2_masscut4.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Trimuon2_Upsilon5_Muon_v*")
#Trimuon2_masscut4.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2p2_DoubleMu5_2p5_Mass5to17")
Trimuon2_masscut4.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Trimuon2_Upsilon5_Muon_NoL1Mass_v*")
#Trimuon2_masscut4.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2")
Trimuon2_masscut4.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Trimuon2_masscut4.DMSelection_ref = cms.string("M<17 & M>5 & Pt>6 & abs(Eta)<2.4")



Trimuon2_masscut5 = hltBPHmonitoring.clone()
Trimuon2_masscut5.FolderName = cms.string('HLT/BPH/DoubleMu3_Trk_L1_masscut5')
Trimuon2_masscut5.tnp = cms.bool(False)
Trimuon2_masscut5.enum = cms.int32(5)
Trimuon2_masscut5.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_Trk_Tau3mu_v*")
#Trimuon2_masscut5.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu5_3_SQ_Mass_Max9")
Trimuon2_masscut5.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v*")
#Trimuon2_masscut5.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2")
Trimuon2_masscut5.muoSelection_ref = cms.string("pt>4 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Trimuon2_masscut5.DMSelection_ref = cms.string("M<9 & abs(Eta)<2.4")



Trimuon2_masscut6 = hltBPHmonitoring.clone()
Trimuon2_masscut6.FolderName = cms.string('HLT/BPH/DoubleMu3_Trk_L1_masscut6')
Trimuon2_masscut6.tnp = cms.bool(False)
Trimuon2_masscut6.enum = cms.int32(5)
Trimuon2_masscut6.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi3p5_Muon2_v*")
#Trimuon2_masscut6.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu5_3_SQ_Mass_Max9")
Trimuon2_masscut6.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_v*")
#Trimuon2_masscut6.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ_OS")
Trimuon2_masscut6.muoSelection_ref = cms.string("pt>3 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Trimuon2_masscut6.DMSelection_ref = cms.string("M<9 & abs(Eta)<2.4")
Trimuon2_masscut6.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32( 30 ),
  xmin  = cms.double(0.),
  xmax  = cms.double(1.5),
)

###triple muon

Dimuon0_tripleMu1 = hltBPHmonitoring.clone()
Dimuon0_tripleMu1.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu1/')
Dimuon0_tripleMu1.tnp = cms.bool(False)
Dimuon0_tripleMu1.enum = cms.int32(6)
Dimuon0_tripleMu1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Trimuon2_Upsilon5_Muon_NoL1Mass_v*")
#Dimuon0_tripleMu1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2")
Dimuon0_tripleMu1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v*")
#Dimuon0_tripleMu1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2")
Dimuon0_tripleMu1.muoSelection_ref = cms.string("pt>4 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu1.DMSelection_ref = cms.string("abs(Eta)<2.4")



Dimuon0_tripleMu2 = hltBPHmonitoring.clone()
Dimuon0_tripleMu2.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu2/')
Dimuon0_tripleMu2.tnp = cms.bool(False)
Dimuon0_tripleMu2.enum = cms.int32(6)
Dimuon0_tripleMu2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Trimuon5_3p5_2_Upsilon_Muon_v*")
#Dimuon0_tripleMu2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu5_3p5_2")
Dimuon0_tripleMu2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v*")
#Dimuon0_tripleMu2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu0")
Dimuon0_tripleMu2.muoSelection_ref = cms.string("pt>4 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu2.DMSelection_ref = cms.string("abs(Eta)<2.4")


Dimuon0_tripleMu3 = hltBPHmonitoring.clone()
Dimuon0_tripleMu3.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_tripleMu3/')
Dimuon0_tripleMu3.tnp = cms.bool(False)
Dimuon0_tripleMu3.enum = cms.int32(6)
Dimuon0_tripleMu3.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v*")
#Dimuon0_tripleMu3.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu_5SQ_3SQ_0OQ")
Dimuon0_tripleMu3.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_LowMass_L1_TM530_v*")
#Dimuon0_tripleMu3.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_TripleMu_5SQ_3SQ_0OQ")
Dimuon0_tripleMu3.muoSelection_ref = cms.string("pt>4 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tripleMu3.DMSelection_ref = cms.string("abs(Eta)<2.4")



###photon 
Dimuon0_photon1 = hltBPHmonitoring.clone()
Dimuon0_photon1.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_photon1/')
Dimuon0_photon1.tnp = cms.bool(False)
Dimuon0_photon1.enum = cms.int32(7)
Dimuon0_photon1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_Photon23_v*")
#Dimuon0_photon1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_EG12")
Dimuon0_photon1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v*")
#Dimuon0_photon1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_EG12")
Dimuon0_photon1.muoSelection_ref = cms.string("pt>10 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_photon1.DMSelection_ref = cms.string("M<30 && abs(Eta)<2.4")


Dimuon0_photon2 = hltBPHmonitoring.clone()
Dimuon0_photon2.FolderName = cms.string('HLT/BPH/DiMu0_Lowmass_L1_photon2/')
Dimuon0_photon2.tnp = cms.bool(False)
Dimuon0_photon2.enum = cms.int32(7)
Dimuon0_photon2.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v*")
#Dimuon0_photon2.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_EG12")
Dimuon0_photon2.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu20_7_Mass0to30_L1_DM4_v*")
#Dimuon0_photon2.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS")
Dimuon0_photon2.muoSelection_ref = cms.string("pt>10 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_photon2.DMSelection_ref = cms.string("M<30 && abs(Eta)<2.4")

###
###L3 TnP
Dimuon0_L3TnP_Jpsi = hltBPHmonitoring.clone()
Dimuon0_L3TnP_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_L1_L3TnP_Jpsi/')
Dimuon0_L3TnP_Jpsi.tnp = cms.bool(True)
Dimuon0_L3TnP_Jpsi.L3 = cms.int32(1)
Dimuon0_L3TnP_Jpsi.enum = cms.int32(1)
##Dimuon0_L3TnP_Jpsi.Jpsi = cms.int32(1)
Dimuon0_L3TnP_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Jpsi_v*")
#Dimuon0_L3TnP_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_L3TnP_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Jpsi_v*")
#Dimuon0_L3TnP_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_L3TnP_Jpsi.muoSelection_ref = cms.string("pt>7.5 &  abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_L3TnP_Jpsi.muoSelection = cms.string("pt>7.5 & abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

Dimuon0_L3TnP_Upsilon = hltBPHmonitoring.clone()
Dimuon0_L3TnP_Upsilon.FolderName = cms.string('HLT/BPH/DiMu0_L1_L3TnP_Upsilon/')
Dimuon0_L3TnP_Upsilon.tnp = cms.bool(True)
Dimuon0_L3TnP_Upsilon.L3 = cms.int32(1)
##Dimuon0_L3TnP_Upsilon.Upsilon = cms.int32(1)
Dimuon0_L3TnP_Upsilon.enum = cms.int32(1)
Dimuon0_L3TnP_Upsilon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Upsilon_v*")
#Dimuon0_L3TnP_Upsilon.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_L3TnP_Upsilon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu7p5_L2Mu2_Upsilon_v*")
#Dimuon0_L3TnP_Upsilon.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_L3TnP_Upsilon.muoSelection_ref = cms.string("pt>7.5 &  abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_L3TnP_Upsilon.muoSelection = cms.string("pt>7.5 & abs(eta)<2.4 & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
###
###HLT OS 

Dimuon0_HLT_OS = hltBPHmonitoring.clone()
Dimuon0_HLT_OS.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_HLT_OS/')
Dimuon0_HLT_OS.tnp = cms.bool(False)
Dimuon0_HLT_OS.enum = cms.int32(2)
Dimuon0_HLT_OS.Jpsi = cms.int32(1)
Dimuon0_HLT_OS.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v*")
#Dimuon0_HLT_OS.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ_OS")
Dimuon0_HLT_OS.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v*")
#Dimuon0_HLT_OS.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_HLT_OS.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_HLT_OS.DMSelection_ref = cms.string("abs(Eta)<2.4")

Dimuon0_HLT_OS1 = hltBPHmonitoring.clone()
Dimuon0_HLT_OS1.FolderName = cms.string('HLT/BPH/DiMu0_Jpsi_L1_HLT_OS1/')
Dimuon0_HLT_OS1.tnp = cms.bool(False)
Dimuon0_HLT_OS1.enum = cms.int32(2)
Dimuon0_HLT_OS1.Jpsi = cms.int32(1)
Dimuon0_HLT_OS1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*")
#Dimuon0_HLT_OS1.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_HLT_OS1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v*")
#Dimuon0_HLT_OS1.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu0_SQ")
Dimuon0_HLT_OS1.muoSelection_ref = cms.string("abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_HLT_OS1.DMSelection_ref = cms.string("abs(Eta)<2.4")


###
###Loose vertex Jpsi
Dimuon0_looseVtx_Jpsi = hltBPHmonitoring.clone()
Dimuon0_looseVtx_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_L1_looseVtx_Jpsi/')
Dimuon0_looseVtx_Jpsi.tnp = cms.bool(False)
Dimuon0_looseVtx_Jpsi.enum = cms.int32(8)
Dimuon0_looseVtx_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v*")
Dimuon0_looseVtx_Jpsi.Jpsi = cms.int32(1)
#Dimuon0_looseVtx_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_looseVtx_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v*")
#Dimuon0_looseVtx_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_looseVtx_Jpsi.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_looseVtx_Jpsi.DMSelection_ref = cms.string("abs(Eta)<1.5")

####Loose vtx Upsilon
Dimuon0_looseVtx_Upsilon = hltBPHmonitoring.clone()
Dimuon0_looseVtx_Upsilon.FolderName = cms.string('HLT/BPH/DiMu0_L1_looseVtx_Upsilon/')
Dimuon0_looseVtx_Upsilon.tnp = cms.bool(False)
Dimuon0_looseVtx_Upsilon.enum = cms.int32(8)
Dimuon0_looseVtx_Upsilon.Upsilon = cms.int32(1)
Dimuon0_looseVtx_Upsilon.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v*")
#Dimuon0_looseVtx_Upsilon.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon0_looseVtx_Upsilon.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Upsilon_NoVertexing_v*")
#Dimuon0_looseVtx_Upsilon.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18")
Dimuon0_looseVtx_Upsilon.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.0 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_looseVtx_Upsilon.DMSelection_ref = cms.string("abs(Eta)<2.0")

###tight vtx
Dimuon0_tightVtx_Jpsi = hltBPHmonitoring.clone()
Dimuon0_tightVtx_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_L1_tightVtx_Jpsi/')
Dimuon0_tightVtx_Jpsi.tnp = cms.bool(False)
Dimuon0_tightVtx_Jpsi.Jpsi = cms.int32(1)
Dimuon0_tightVtx_Jpsi.enum = cms.int32(8)
Dimuon0_tightVtx_Jpsi.displaced = cms.int32(1)
Dimuon0_tightVtx_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_Displaced_v*")
#Dimuon0_tightVtx_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_tightVtx_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_NoVertexing_v*")
#Dimuon0_tightVtx_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_tightVtx_Jpsi.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_tightVtx_Jpsi.DMSelection_ref = cms.string("abs(Eta)<2.4")


###additional track

Dimuon0_addTrack_Jpsi = hltBPHmonitoring.clone()
Dimuon0_addTrack_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_L1_addTrack_Jpsi/')
Dimuon0_addTrack_Jpsi.tnp = cms.bool(False)
Dimuon0_addTrack_Jpsi.enum = cms.int32(9)
Dimuon0_addTrack_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_Displaced_v*")
#Dimuon0_addTrack_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrack_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_Displaced_v*")
#Dimuon0_addTrack_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrack_Jpsi.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_addTrack_Jpsi.DMSelection_ref = cms.string("abs(Eta)<2.4")
Dimuon0_addTrack_Jpsi.trOrMu = cms.int32(True)

Dimuon0_addTrackTrack_Jpsi = hltBPHmonitoring.clone()
Dimuon0_addTrackTrack_Jpsi.FolderName = cms.string('HLT/BPH/DiMu0_L1_addTrackTrack_Jpsi/')
Dimuon0_addTrackTrack_Jpsi.tnp = cms.bool(False)
Dimuon0_addTrackTrack_Jpsi.trOrMu = cms.int32(True)
Dimuon0_addTrackTrack_Jpsi.enum = cms.int32(11)
Dimuon0_addTrackTrack_Jpsi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrkTrk_Displaced_v*")
#Dimuon0_addTrackTrack_Jpsi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrackTrack_Jpsi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_Jpsi_Displaced_v*")
#Dimuon0_addTrackTrack_Jpsi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
Dimuon0_addTrackTrack_Jpsi.muoSelection_ref = cms.string("pt>5 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_addTrackTrack_Jpsi.DMSelection_ref = cms.string("abs(Eta)<2.4")

DM2_Jpsi_addTrackTrack_Phi = hltBPHmonitoring.clone()
DM2_Jpsi_addTrackTrack_Phi.FolderName = cms.string('HLT/BPH/DM2_Jpsi_addTrackTrack_Phi/')
DM2_Jpsi_addTrackTrack_Phi.tnp = cms.bool(False)
DM2_Jpsi_addTrackTrack_Phi.enum = cms.int32(11)
DM2_Jpsi_addTrackTrack_Phi.Jpsi = cms.int32(1)
DM2_Jpsi_addTrackTrack_Phi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v*")
#DM2_Jpsi_addTrackTrack_Phi.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
DM2_Jpsi_addTrackTrack_Phi.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon0_Jpsi_NoVertexing_v*")
#DM2_Jpsi_addTrackTrack_Phi.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4")
#DM2_Jpsi_addTrackTrack_Phi.muoSelection = cms.string("pt>2 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DM2_Jpsi_addTrackTrack_Phi.muoSelection_ref = cms.string("pt>2 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
DM2_Jpsi_addTrackTrack_Phi.minprob = cms.double(0.1)
DM2_Jpsi_addTrackTrack_Phi.mincos = cms.double(0.)
DM2_Jpsi_addTrackTrack_Phi.minDS = cms.double(0.)
DM2_Jpsi_addTrackTrack_Phi.trOrMu = cms.int32(True)
DM2_Jpsi_addTrackTrack_Phi.trSelection_ref = cms.string("pt>1")
DM2_Jpsi_addTrackTrack_Phi.minmassTkTk = cms.double(0.920)
DM2_Jpsi_addTrackTrack_Phi.maxmassTkTk = cms.double(1.120)
DM2_Jpsi_addTrackTrack_Phi.minmassJpsiTk = cms.double(0.)
DM2_Jpsi_addTrackTrack_Phi.maxmassJpsiTk = cms.double(99.)
DM2_Jpsi_addTrackTrack_Phi.histoPSet.ptBinning = [-0.5, 0, 0.5, 1, 1.5, 2, 4, 8, 10, 12, 14, 16, 18, 20]
DM2_Jpsi_addTrackTrack_Phi.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32( 32 ),
  xmin  = cms.double(-3.2),
  xmax  = cms.double( 3.2),
)

DM2_Jpsi_addTkMuTkMu_Phi = DM2_Jpsi_addTrackTrack_Phi.clone()
DM2_Jpsi_addTkMuTkMu_Phi.FolderName = cms.string('HLT/BPH/DM2_Jpsi_addTkMuTkMu_Phi/')
DM2_Jpsi_addTkMuTkMu_Phi.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu2_Jpsi_DoubleTkMu0_Phi_v*")
DM2_Jpsi_addTkMuTkMu_Phi.trSelection_ref = cms.string("")


Dimuon0_addTrackMu_Onia = hltBPHmonitoring.clone()
Dimuon0_addTrackMu_Onia.FolderName = cms.string('HLT/BPH/DiMu0_L1_addTrackMu_Onia/')
Dimuon0_addTrackMu_Onia.tnp = cms.bool(False)
Dimuon0_addTrackMu_Onia.minmassJpsiTk = cms.double(3)
Dimuon0_addTrackMu_Onia.maxmassJpsiTk = cms.double(3.2)
Dimuon0_addTrackMu_Onia.enum = cms.int32(10)
Dimuon0_addTrackMu_Onia.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu30_TkMu0_Onia_v*")
#Dimuon0_addTrackMu_Onia.numGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_SingleMu22 OR L1_SingleMu25")
Dimuon0_addTrackMu_Onia.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu27_v*")
#Dimuon0_addTrackMu_Onia.denGenericTriggerEventPSet.l1Algorithms = cms.vstring("L1_SingleMu22 OR L1_SingleMu25")
Dimuon0_addTrackMu_Onia.muoSelection_ref = cms.string("pt>32 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_addTrackMu_Onia.DMSelection_ref = cms.string("abs(Eta)<2.4")
Dimuon0_addTrackMu_Onia.trOrMu = cms.int32(True)

Dimuon0_addTrackMu_Phi1 = hltBPHmonitoring.clone()
Dimuon0_addTrackMu_Phi1.FolderName = cms.string('HLT/BPH/DiMu0_L1_addTrackMu_Phi1/')
Dimuon0_addTrackMu_Phi1.tnp = cms.bool(False)
Dimuon0_addTrackMu_Phi1.minmassJpsiTk = cms.double(0.920)
Dimuon0_addTrackMu_Phi1.maxmassJpsiTk = cms.double(1.120)
Dimuon0_addTrackMu_Phi1.enum = cms.int32(10)
Dimuon0_addTrackMu_Phi1.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu25_TkMu0_Phi_v*")
Dimuon0_addTrackMu_Phi1.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu20_v*")
Dimuon0_addTrackMu_Phi1.muoSelection_ref = cms.string("pt>26 && abs(eta)<2.4 &  isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
Dimuon0_addTrackMu_Phi1.DMSelection_ref = cms.string("abs(Eta)<2.4")
Dimuon0_addTrackMu_Phi1.trOrMu = cms.int32(True)

DimuonX_HLT_OS_Vtx = hltBPHmonitoring.clone()
DimuonX_HLT_OS_Vtx.FolderName = cms.string('HLT/BPH/DimuX_HLT_OS_Vtx/')
DimuonX_HLT_OS_Vtx.tnp = cms.bool(False)
DimuonX_HLT_OS_Vtx.enum = cms.int32(2)
DimuonX_HLT_OS_Vtx.Jpsi = cms.int32(1)
DimuonX_HLT_OS_Vtx.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Dimuon0_Jpsi_L1_NoOS_v*')
#DimuonX_HLT_OS_Vtx.numGenericTriggerEventPSet.l1Algorithms = cms.vstring('L1_DoubleMu0_SQ')
DimuonX_HLT_OS_Vtx.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v*')
#DimuonX_HLT_OS_Vtx.denGenericTriggerEventPSet.l1Algorithms = cms.vstring('L1_DoubleMu0_SQ')
DimuonX_HLT_OS_Vtx.muoSelection_ref = cms.string('abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0')

#hltBPHmonitoring.histoPSet.ptBinning = [-0.5, 0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 50, 70]
#hltBPHmonitoring.histoPSet.dMuPtBinning = [-0.5, 0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 50, 70]

###

bphHLTmonitoring = cms.Sequence(
    Dimuon25_Jpsi_tnp
    + Dimuon0_Jpsi_tnp
    + Dimuon0_Jpsi_tnp1
    + Dimuon0_Jpsi_tnp2
    + Dimuon0_Jpsi_OS
    + Dimuon0_er
    + Dimuon0_Upsilon_er
    + DMu4_3_Bs_dRcut
    + DMu4_3_Jpsi_dRcut
    + Dimuon14_Phi_dRcut
#    + DMu4_LowMassNonResonantTrk_Displaced_dRcut
#    + DMu4_JpsiTrk_Displaced_dRcut
#    + DMu4_PsiPrimeTrk_Displaced_dRcut
    + Dimuon25_Jpsi_dRcut
    + Dimuon18_PsiPrime_dRcut
    + Dimuon25_Jpsi_dRcut_low
    + Dimuon18_PsiPrime_dRcut_low
#    + DMu4_PsiPrimeTrk_Displaced_dRcut_low
#    + DMu4_JpsiTrk_Displaced_dRcut_low
#    + DMu4_LowMassNonResonantTrk_Displaced_dRcut_low
    + Dimuon20_masscut1
    + Dimuon12_masscut2
    + Trimuon2_masscut4
    + Trimuon2_masscut5
    + Trimuon2_masscut6
    + Dimuon0_tripleMu1
    + Dimuon0_tripleMu2
    + Dimuon0_tripleMu3
#    + Dimuon0_photon1
#    + Dimuon0_photon2
    + Dimuon0_L3TnP_Jpsi
    + Dimuon0_L3TnP_Upsilon
    + Dimuon0_HLT_OS
    + Dimuon0_HLT_OS1
    + Dimuon0_looseVtx_Jpsi
    + Dimuon0_looseVtx_Upsilon
    + Dimuon0_tightVtx_Jpsi
    + Dimuon0_addTrack_Jpsi
    + Dimuon0_addTrackTrack_Jpsi
    + DM2_Jpsi_addTrackTrack_Phi
    + DM2_Jpsi_addTkMuTkMu_Phi
    + Dimuon0_addTrackMu_Onia
    + Dimuon0_addTrackMu_Phi1
    #+ DimuonX_HLT_OS_Vtx
)



bphMonitorHLT = cms.Sequence(
    bphHLTmonitoring * 
    tau3MuMonitorHLT    
)

bphHLTDQMSourceExtra = cms.Sequence(
)

