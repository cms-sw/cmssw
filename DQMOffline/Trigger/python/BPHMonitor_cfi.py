import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.bphMonitoring_cfi import bphMonitoring

hltBPHmonitoring = bphMonitoring.clone()
hltBPHmonitoring.FolderName = cms.string('HLT/BPH/Dimuon_10_Jpsi_Barrel/')
hltBPHmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.int32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(200),
)
hltBPHmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.int32 (  64  ),
  xmin  = cms.double(   -3.2),
  xmax  = cms.double(3.2),
)
hltBPHmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.int32 (  24  ),
  xmin  = cms.double(   -2.4),
  xmax  = cms.double(2.4),
)
hltBPHmonitoring.histoPSet.d0PSet = cms.PSet(
  nbins = cms.int32 (  200  ),
  xmin  = cms.double(   -5.),
  xmax  = cms.double(5),
)
hltBPHmonitoring.histoPSet.z0PSet = cms.PSet(
  nbins = cms.int32 (  300 ),
  xmin  = cms.double(   -15),
  xmax  = cms.double(15),
)

hltBPHmonitoring.tracks       = cms.InputTag("generalTracks") # tracks??
hltBPHmonitoring.offlinePVs      = cms.InputTag("offlinePrimaryVertices") # PVs
hltBPHmonitoring.beamSpot = cms.InputTag("offlineBeamSpot") # 
hltBPHmonitoring.muons     = cms.InputTag("muons") # 

hltBPHmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBPHmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("BPHDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltBPHmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltBPHmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltBPHmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_Dimuon10_Jpsi_Barrel") # HLT_ZeroBias_v*
hltBPHmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("diMu10")
hltBPHmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltBPHmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltBPHmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltBPHmonitoring.denGenericTriggerEventPSet.hltPaths  = cms.vstring( "HLT_Dimuon6_Jpsi_NoVertexing" )#reference
hltBPHmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 0,1,2,3,5,6,7,8,9,12,13,14,15,16,17,20,22,24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !TODO
hltBPHmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltBPHmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

