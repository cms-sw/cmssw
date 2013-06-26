import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi


ecalpi0CalibHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
# HLTPaths = ['AlCa_EcalPi0','AlCa_EcalEta'],
  eventSetupPathsKey='EcalCalPi0Calib',   
  throw = False
  )
 
