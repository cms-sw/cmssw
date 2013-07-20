import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.FWLiteParams = cms.PSet(
  input    = cms.string('file:patTuple.root'),
  jetSrc   = cms.InputTag('cleanPatJets'),
  overlaps = cms.string('electrons')
)
