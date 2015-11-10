import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContentHeavyIons_cff import FEVTEventContent, RECOEventContent, AODEventContent

skimFEVTContent = FEVTEventContent.clone()
skimFEVTContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimFEVTContent.outputCommands.append("drop *_*_*_SKIM")

skimRECOContent = RECOEventContent.clone()
skimRECOContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRECOContent.outputCommands.append("drop *_*_*_SKIM")

skimAODContent = AODEventContent.clone()
skimAODContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimAODContent.outputCommands.append("drop *_*_*_SKIM")

#####################

from Configuration.Skimming.HI_PhotonSkim_cff import *
photonSkimPath = cms.Path( photonSkimSequence )
SKIMStreamPhoton = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'Photon',
    paths = (photonSkimPath),
    content = skimAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )


#####################

from Configuration.Skimming.HI_ZEESkim_cff import *
zEESkimPath = cms.Path( zEESkimSequence )
SKIMStreamZEE = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'ZEE',
    paths = (zEESkimPath),
    content = skimAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

#####################

from Configuration.Skimming.HI_ZMMSkim_cff import *
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

from Configuration.Skimming.HI_BJetSkim_cff import *
bJetSkimPath = cms.Path( bJetSkimSequence )
SKIMStreamBJet = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'BJet',
    paths = (bJetSkimPath),
    content = skimAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

#####################

from Configuration.Skimming.HI_D0MesonSkim_cff import *
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

from Configuration.Skimming.HI_HighPtJetSkim_cff import *
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

from Configuration.Skimming.HI_OniaCentralSkim_cff import *
oniaCentralSkimPath = cms.Path( oniaCentralSkimSequence )
SKIMStreamOniaCentral = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'OniaCentral',
    paths = (oniaCentralSkimPath),
    content = skimRECOContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

#####################     

from Configuration.Skimming.HI_OniaPeripheralSkim_cff import *
oniaPeripheralSkimPath = cms.Path( oniaPeripheralSkimSequence )
SKIMStreamOniaPeripheral = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'OniaPeripheral',
    paths = (oniaPeripheralSkimPath),
    content = skimRECOContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RECO')
    )

#####################     

from Configuration.Skimming.HI_SingleTrackSkim_cff import *
singleTrackSkimPath = cms.Path( singleTrackSkimSequence )
SKIMStreamSingleTrack = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'SingleTrack',
    paths = (singleTrackSkimPath),
    content = skimAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

#####################     

from Configuration.Skimming.HI_MinBiasSkim_cff import *
minBiasSkimPath = cms.Path( minBiasSkimSequence )
SKIMStreamMinBias = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'MinBias',
    paths = (minBiasSkimPath),
    content = skimAODContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

#####################      


from Configuration.Skimming.HI_OniaUPCSkim_cff import *
oniaUPCSkimPath = cms.Path( oniaUPCSkimSequence )
SKIMStreamOniaUPC = cms.FilteredStream(
    responsible = 'HI PAG',
    name = 'OniaUPC',
    paths = (oniaUPCSkimPath),
    content = skimFEVTContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
    )

#####################      
