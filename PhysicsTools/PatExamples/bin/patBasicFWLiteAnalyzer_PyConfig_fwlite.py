import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")


process.FWLiteParams = cms.PSet(
    muonSrc = cms.InputTag('cleanPatMuons')
)

