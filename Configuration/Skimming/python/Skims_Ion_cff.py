import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import MINIAODEventContent

skimMINIAODEventContent = MINIAODEventContent.clone()
skimMINIAODEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimMINIAODEventContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.Skimming.Ion_MuonSkim_cff import *

HighPtMuonIonSkimPath = cms.Path( HighPtMuonIonSkimSequence )
SKIMStreamIonHighPtMuon = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'IonHighPtMuon',
    paths = (HighPtMuonIonSkimPath),
    content = skimMINIAODEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

DimuonIonSkimPath = cms.Path( DimuonIonSkimSequence )
SKIMStreamIonDimuon = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'IonDimuon',
    paths = (DimuonIonSkimPath),
    content = skimMINIAODEventContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('USER')
    )

#####################
