import FWCore.ParameterSet.Config as cms
 
ecalBadCalibSep2017ListFilter = cms.EDFilter(
  "ecalBadCalibSep2017ListFilter",
 
  # use this if using AOD:
  # the EB rechit collection found in AOD
  EBRecHitSource = cms.InputTag('reducedEcalRecHitsEB'), 
  # the EE rechit collection found in AOD
  EERecHitSource = cms.InputTag('reducedEcalRecHitsEE'),
 
 
  # use this if using MINIAOD:
  # the EB rechit collection found in MINIAOD
  # EBRecHitSource = cms.InputTag('reducedEgamma','reducedEBRecHits'), 
  # the EE rechit collection found in MINIAOD
  # EERecHitSource = cms.InputTag('reducedEgamma','reducedEERecHits'),
 
 
 
  # minimum rechit et to flag as bad:  EB 
  ebMinEt        = cms.double(50.),
  # minimum rechit et to flag as bad:  EE 
  eeMinEt        = cms.double(50.),
  # DetId of bad channel: EE
  baddetEB        = cms.vuint32(0),
  # DetId of bad channel: EE
  baddetEE        = cms.vuint32(872439604,872422825,872420274,872423218,
                                872423215,872416066,872435036,872439336,
                                872420273,872436907,872420147,872439731,
                                872436657,872420397,872439732,872439339,
                                872439603),                                  
  taggingMode = cms.bool(False),
  #prints debug info for each channel if set to true
  debug = cms.bool(False),
)
