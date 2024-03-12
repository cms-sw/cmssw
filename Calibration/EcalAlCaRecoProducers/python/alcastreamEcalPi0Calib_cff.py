import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi


ecalpi0CalibHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
# HLTPaths = ['AlCa_EcalPi0','AlCa_EcalEta'],
  eventSetupPathsKey='EcalCalPi0Calib',   
  throw = False
  )
 
# foo bar baz
# lK7JCvU2VQog7
# 0H9D0uUqjIDrz
