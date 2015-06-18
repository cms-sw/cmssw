import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent

OutALCARECOMETHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "pathHotlineSkimPFMET",
            "pathHotlineSkimCaloMET",
            "pathHotlineSkimCondMET",
        ),
    ),
    outputCommands = FEVTEventContent.outputCommands 
)
