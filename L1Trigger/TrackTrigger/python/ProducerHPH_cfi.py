import FWCore.ParameterSet.Config as cms

# ParameterSet used by HitPatternHelper

HitPatternHelper_params = cms.PSet (

  hphDebug   = cms.bool(False),   
  useNewKF   = cms.bool(False),
  deltaTanL  = cms.double(0.125)

)
