import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import RECOEventContent, FEVTEventContent

skimRECOContent = RECOEventContent.clone()
skimRECOContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRECOContent.outputCommands.append("drop *_*_*_SKIM")

skimFEVTContent = FEVTEventContent.clone()
skimFEVTContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimFEVTContent.outputCommands.append("drop *_*_*_SKIM")

#####################

from Configuration.Skimming.PP_HighPtJetSkim_cff import *
highPtJetSkimPath = cms.Path( highPtJetSkimSequence )
SKIMStreamHighPtJet = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'HighPtJet',
    paths = (highPtJetSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from Configuration.Skimming.PP_ZMMSkim_cff import *
zMMSkimPath = cms.Path( zMMSkimSequence )
SKIMStreamZMM = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'ZMM',
    paths = (zMMSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################

from Configuration.Skimming.PP_OniaSkim_cff import *
oniaSkimPath = cms.Path( oniaSkimSequence )
SKIMStreamOnia = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'Onia',
    paths = (oniaSkimPath),
    content = skimRECOContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

#####################

from Configuration.Skimming.PP_D0MesonSkim_cff import *
d0MesonSkimPath = cms.Path( d0MesonSkimSequence )
SKIMStreamD0Meson = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'D0Meson',
    paths = (d0MesonSkimPath),
    content = skimRECOContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

#####################
