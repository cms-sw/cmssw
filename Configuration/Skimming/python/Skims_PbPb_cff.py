import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent

skimFEVTContent = FEVTEventContent.clone()
skimFEVTContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimFEVTContent.outputCommands.append("drop *_*_*_SKIM")


#####################


from Configuration.Skimming.PbPb_EMuSkim_cff import *
emuSkimPath = cms.Path( emuSkimSequence )
SKIMStreamPbPbEMu = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PbPbEMu',
    paths = (emuSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PbPb_ZEESkim_cff import *
zEESkimPath = cms.Path( zEESkimSequence )
SKIMStreamPbPbZEE = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PbPbZEE',
    paths = (zEESkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      


from Configuration.Skimming.PbPb_ZMMSkim_cff import *
zMMSkimPath = cms.Path( zMMSkimSequence )
SKIMStreamPbPbZMM = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PbPbZMM',
    paths = (zMMSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      

from Configuration.Skimming.PbPb_ZMuSkimMuonDPG_cff import *
ZMuSkimPathPbPb = cms.Path( diMuonSelSeqForPbPbZMuSkim )
SKIMStreamPbPbZMu = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'PbPbZMu',
    paths = (ZMuSkimPathPbPb),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      
