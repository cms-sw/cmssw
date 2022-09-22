import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackFindingTMTT.TMTrackProducer_Defaults_cfi import TMTrackProducer_params

# ParameterSet used by HitPatternHelper

HitPatternHelper_params = cms.PSet (

  hphDebug   = cms.bool(False), # switch on/off debug statement
  useNewKF   = cms.bool(False), # switch between new/old KF
  oldKFPSet  = cms.PSet(TMTrackProducer_params.EtaSectors) # import eta sector boundries from old kf package
  
)
