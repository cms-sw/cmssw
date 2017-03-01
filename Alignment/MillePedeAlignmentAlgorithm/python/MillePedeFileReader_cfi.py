import FWCore.ParameterSet.Config as cms

MillePedeFileReader = cms.PSet(
  millePedeLogFile = cms.string('millepede.log'),
  millePedeResFile = cms.string('millepede.res'),

  # signifiance of movement must be above
  sigCut = cms.double(2.5),

  # cutoff in micro-meter & micro-rad
  Xcut  = cms.double( 5.0),
  tXcut = cms.double(30.0), # thetaX
  Ycut  = cms.double(10.0),
  tYcut = cms.double(30.0), # thetaY
  Zcut  = cms.double(15.0),
  tZcut = cms.double(30.0), # thetaZ

  # maximum movement in micro-meter/rad
  maxMoveCut  = cms.double(200.0),
  maxErrorCut = cms.double( 10.0)
)
