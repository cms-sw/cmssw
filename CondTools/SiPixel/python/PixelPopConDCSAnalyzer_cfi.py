import FWCore.ParameterSet.Config as cms

PixelCaenChannelIsOn = cms.EDAnalyzer("PixelPopConBoolAnalyzer",

  record = cms.string('PixelCaenChannelIsOnRcd'),
  loggingOn = cms.untracked.bool(True),
  SinceAppendMode = cms.bool(True),

  Source = cms.PSet
  (
    authenticationPath = cms.string('.'),
    dbName = cms.string('CMSONR'),
    tables = cms.vstring('DCSLASTVALUE_ON')
  )
)

PixelCaenChannelIMon = cms.EDAnalyzer("PixelPopConFloatAnalyzer",

  record = cms.string('PixelCaenChannelIMonRcd'),
  loggingOn = cms.untracked.bool(True),
  SinceAppendMode = cms.bool(True),

  Source = cms.PSet
  (
    authenticationPath = cms.string('.'),
    dbName = cms.string('CMSONR'),
    tables = cms.vstring('DCSLASTVALUE_CURRENT')
  )
)

PixelCaenChannel = cms.EDAnalyzer("PixelPopConCaenChannelAnalyzer",

  record = cms.string('PixelCaenChannelRcd'),
  loggingOn = cms.untracked.bool(True),
  SinceAppendMode = cms.bool(True),

  Source = cms.PSet
  (
    authenticationPath = cms.string('.'),
    dbName = cms.string('CMSONR'),
    tables = cms.vstring('DCSLASTVALUE_ON', 'DCSLASTVALUE_CURRENT', 'DCSLASTVALUE_VOLTAGE')
  )
)
