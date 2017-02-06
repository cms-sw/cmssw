import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent

skimFEVTContent = FEVTEventContent.clone()
skimFEVTContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimFEVTContent.outputCommands.append("drop *_*_*_SKIM")


#####################


from Configuration.Skimming.PA_MinBiasSkim_cff import *
minBiasSkimPath = cms.Path( minBiasSkimSequence )
SKIMStreamPAMinBias = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAMinBias',
    paths = (minBiasSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PA_ZEESkim_cff import *
zEESkimPath = cms.Path( zEESkimSequence )
SKIMStreamPAZEE = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAZEE',
    paths = (zEESkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PA_ZMMSkim_cff import *
zMMSkimPath = cms.Path( zMMSkimSequence )
SKIMStreamPAZMM = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAZMM',
    paths = (zMMSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      
