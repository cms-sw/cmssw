import FWCore.ParameterSet.Config as cms

OutALCARECOMETHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "pathHotlineSkimPFMET",
            "pathHotlineSkimCaloMET",
            "pathHotlineSkimCondMET",
        ),
    ),
    outputCommands = cms.untracked.vstring(
        'keep *'
        )
)
