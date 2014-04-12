import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi


ecaletaCalibHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#  HLTPaths = ['AlCa_EcalEta'],
  eventSetupPathsKey='EcalCalEtaCalib',   
  throw = False
  )
 
