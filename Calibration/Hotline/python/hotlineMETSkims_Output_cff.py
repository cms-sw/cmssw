import FWCore.ParameterSet.Config as cms

OutALCARECOMETHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "hotlineSkimPFMET",
            "hotlineSkimCaloMET",
            "hotlineSkimCondMET"
        ),
    ),
    outputCommands = cms.untracked.vstring('keep *')
)
