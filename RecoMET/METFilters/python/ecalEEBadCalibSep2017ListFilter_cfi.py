import FWCore.ParameterSet.Config as cms
 
ecalBadCalibSep2017ListFilter = cms.EDFilter(
  "EcalBadCalibFilter",
 
  # use this if using AOD:
  # the Ecal rechit collection found in AOD
  EERecHitSource = cms.InputTag('reducedEcalRecHitsEE'),
 
 
  # use this if using MINIAOD:
  # the Ecal rechit collection found in MINIAOD
  # EERecHitSource = cms.InputTag('reducedEgamma','reducedEERecHits'),
 
 
 
  # minimum rechit et to flag as bad: 
  eeMinEt        = cms.double(50.),
  # DetId of bad channel:
  baddetEE        = cms.vuint32(872439604,872422825,872420274,872423218,
                                872423215,872416066,872435036,872439336,
                                872420273,872436907,872420147,872439731,
                                872436657,872420397,872439732,872439339,
                                872439603),                                  
  taggingMode = cms.bool(False),
  #prints debug info for each channel if set to true
  debug = cms.bool(False),
)
