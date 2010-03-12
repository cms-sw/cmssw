import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Jet30U_Quad_Fwd_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Jet30U_Quad_Fwd_8e29.HLTPaths = ("HLT_Jet30U","HLT_QuadJet15U","HLT_FwdJet20U")
Jet30U_Quad_Fwd_8e29.HLTPathsPrescales  = cms.vuint32(10,1,10)
Jet30U_Quad_Fwd_8e29.HLTOverallPrescale = cms.uint32(1)
Jet30U_Quad_Fwd_8e29.andOr = True

filterSdJet30U_Quad_Fwd_8e29 = cms.Path(Jet30U_Quad_Fwd_8e29)
