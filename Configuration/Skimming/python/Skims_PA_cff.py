import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent

skimFEVTContent = FEVTEventContent.clone()
skimFEVTContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimFEVTContent.outputCommands.append("drop *_*_*_SKIM")


#####################


from Configuration.Skimming.PA_MinBiasSkim_cff import *
minBiasPASkimPath = cms.Path( minBiasPASkimSequence )
SKIMStreamPAMinBias = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAMinBias',
    paths = (minBiasPASkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PA_ZEESkim_cff import *
zEEPASkimPath = cms.Path( zEEPASkimSequence )
SKIMStreamPAZEE = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAZEE',
    paths = (zEEPASkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PA_ZMMSkim_cff import *
zMMPASkimPath = cms.Path( zMMPASkimSequence )
SKIMStreamPAZMM = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PAZMM',
    paths = (zMMPASkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      
