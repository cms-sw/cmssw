import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")


process.FWLiteParams = cms.PSet(
    wSelectorParams = cms.PSet(
        muonSrc = cms.InputTag('cleanPatMuons'),
        metSrc = cms.InputTag('patMETs'),
        muonPtMin = cms.double(20.0),
        metMin = cms.double(20.0),
        cutsToIgnore = cms.vstring('MET')
        )
)

