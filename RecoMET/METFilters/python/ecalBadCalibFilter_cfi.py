import FWCore.ParameterSet.Config as cms
 
ecalBadCalibFilter = cms.EDFilter(
  "EcalBadCalibFilter",
 
  # use this if using AOD:
  # the Ecal rechit collection found in AOD
  EcalRecHitSource = cms.InputTag('reducedEcalRecHitsEE'),
 
 
  # use this if using MINIAOD:
  # the Ecal rechit collection found in MINIAOD
  # EcalRecHitSource = cms.InputTag('reducedEgamma','reducedEERecHits'),
 
 
 
  # minimum rechit et to flag as bad: 
  ecalMinEt        = cms.double(50.),
  # DetId of bad channel:
  baddetEcal       = cms.vuint32(),

  taggingMode = cms.bool(False),
  #prints debug info for each channel if set to true
  debug = cms.bool(False),
)

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify(ecalBadCalibFilter, baddetEcal = [872439604,872422825,872420274,872423218,
                                                       872423215,872416066,872435036,872439336,
                                                       872420273,872436907,872420147,872439731,
                                                       872436657,872420397,872439732,872439339,
                                                       872439603,872422436,872439861,872437051,
                                                       872437052,872420649,872421950,872437185,
                                                       872422564,872421566,872421695,872421955,
                                                       872421567,872437184,872421951,872421694,
                                                       872437056,872437057,872437313])
