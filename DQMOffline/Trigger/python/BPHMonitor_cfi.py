import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.bphMonitoring_cfi import bphMonitoring

hltBPHmonitoring = metMonitoring.clone()
hltBPHmonitoring.FolderName = cms.string('HLT/BPH/Dimuon_10_Jpsi_Barrel/')
hltBPHmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.int32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
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
hltBPHmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltBPHmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltBPHmonitoring.denGenericTriggerEventPSet.hltPaths  = cms.vstring( "HLT_Dimuon6_Jpsi_NoVertexing" )#reference
hltBPHmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !TODO
hltBPHmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltBPHmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltBPHmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

