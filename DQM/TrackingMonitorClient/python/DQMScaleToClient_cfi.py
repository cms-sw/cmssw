import FWCore.ParameterSet.Config as cms

dqmScaleToClient = cms.EDAnalyzer('DQMScaleToClient',
  outputme = cms.PSet(
    folder = cms.string(''),
    name = cms.string(''),
    factor = cms.double(1) # it will scale the me to 1 [h->Scale(1./integral)]
  ),
  inputme = cms.PSet(
    folder = cms.string(''),
    name = cms.string('')
  )
)
