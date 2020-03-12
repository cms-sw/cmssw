import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.bphMonitoring_cfi import bphMonitoring as _bphMonitoring

hltBPHmonitoring = _bphMonitoring.clone()

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltBPHmonitoring, stageL1Trigger = 2)

#hltBPHmonitoring.options = cms.untracked.PSet(
#    SkipEvent = cms.untracked.vstring('ProductNotFound')
#)
hltBPHmonitoring.FolderName = cms.string('HLT/BPH/Dimuon_10_Jpsi_Barrel/')
hltBPHmonitoring.tnp = cms.bool(True)
hltBPHmonitoring.max_dR = cms.double(1.4)
hltBPHmonitoring.minmass = cms.double(2.596)
hltBPHmonitoring.maxmass = cms.double(3.596)
hltBPHmonitoring.Upsilon = cms.int32(0)
hltBPHmonitoring.Jpsi = cms.int32(0)
hltBPHmonitoring.seagull = cms.int32(0)
hltBPHmonitoring.ptCut = cms.int32(0)
hltBPHmonitoring.displaced = cms.int32(0)

hltBPHmonitoring.histoPSet.ptBinning = [-0.5, 0, 2, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50]

hltBPHmonitoring.histoPSet.dMuPtBinning = [6, 8, 12, 16,  20,  25, 30, 35, 40, 50, 70]

hltBPHmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(8),
  xmin  = cms.double(-3.2),
  xmax  = cms.double(3.2),
)
hltBPHmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(12),
  xmin  = cms.double(-2.4),
  xmax  = cms.double(2.4),
)
hltBPHmonitoring.histoPSet.d0PSet = cms.PSet(
  nbins = cms.uint32(50),
  xmin  = cms.double(-5.),
  xmax  = cms.double(5),
)
hltBPHmonitoring.histoPSet.z0PSet = cms.PSet(
  nbins = cms.uint32(60),
  xmin  = cms.double(-15),
  xmax  = cms.double(15),
)

hltBPHmonitoring.histoPSet.dRPSet = cms.PSet(
  nbins = cms.uint32(26),
  xmin  = cms.double(0),
  xmax  = cms.double(1.3),
)

hltBPHmonitoring.histoPSet.massPSet = cms.PSet(
  nbins = cms.uint32(140),
  xmin  = cms.double(0),
  xmax  = cms.double(7.),
)
hltBPHmonitoring.histoPSet.BmassPSet = cms.PSet(
  nbins = cms.uint32(20),
  xmin  = cms.double(5.1),
  xmax  = cms.double(5.5),
)

hltBPHmonitoring.histoPSet.dcaPSet = cms.PSet(
  nbins = cms.uint32(10),
  xmin  = cms.double(0),
  xmax  = cms.double(0.5),
)

hltBPHmonitoring.histoPSet.dsPSet = cms.PSet(
  nbins = cms.uint32(15),
  xmin  = cms.double(0),
  xmax  = cms.double(60),
)

hltBPHmonitoring.histoPSet.cosPSet = cms.PSet(
  nbins = cms.uint32(10),
  xmin  = cms.double(0.9),
  xmax  = cms.double(1),
)

hltBPHmonitoring.histoPSet.probBinning = [0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

hltBPHmonitoring.tracks = cms.InputTag("generalTracks") # tracks??
hltBPHmonitoring.offlinePVs = cms.InputTag("offlinePrimaryVertices") # PVs
hltBPHmonitoring.beamSpot = cms.InputTag("offlineBeamSpot") # 

hltBPHmonitoring.muons = cms.InputTag("muons") # 
hltBPHmonitoring.photons = cms.InputTag("photons") # 
hltBPHmonitoring.hltTriggerSummaryAOD   = cms.InputTag("hltTriggerSummaryAOD","","HLT")
#hltBPHmonitoring.DMSelection_ref = cms.string("")
#hltBPHmonitoring.muoSelection_ref = cms.string("")
#hltBPHmonitoring.muoSelection_ = cms.string("")

hltBPHmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltBPHmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("BPHDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltBPHmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltBPHmonitoring.numGenericTriggerEventPSet.andOrL1      = cms.bool(True)# True:=OR; False:=AND
hltBPHmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT")
hltBPHmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_Dimuon0_Jpsi_L1_NoOS_v*") # HLT_ZeroBias_v*
#hltBPHmonitoring.numGenericTriggerEventPSet.l1Algorithms      = cms.vstring("L1_DoubleMu0_SQ") # HLT_ZeroBias_v*
#hltBPHmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("diMu10")
hltBPHmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltBPHmonitoring.numGenericTriggerEventPSet.errorReplyL1 = cms.bool( True )
hltBPHmonitoring.numGenericTriggerEventPSet.l1BeforeMask = cms.bool( True )
hltBPHmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltBPHmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
#hltBPHmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltBPHmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltBPHmonitoring.denGenericTriggerEventPSet.hltPaths  = cms.vstring( "HLT_Mu7p5_Track2_Jpsi_v*" )#reference
#hltBPHmonitoring.denGenericTriggerEventPSet.l1Algorithms      = cms.vstring("L1_DoubleMu0_SQ") # HLT_ZeroBias_v*
#hltBPHmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 0,1,2,3,5,6,7,8,9,12,13,14,15,16,17,20,22,24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel
hltBPHmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltBPHmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
