import FWCore.ParameterSet.Config as cms

MillePedeFileReader = cms.PSet(
  fileDir = cms.string(''),
  
  millePedeEndFile = cms.string('millepede.end'),
  millePedeLogFile = cms.string('millepede.log'),
  millePedeResFile = cms.string('millepede.res'),
  
  isHG = cms.bool(False)
  )
