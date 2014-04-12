import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

dimuonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_selectedPatMuonsTriggerMatch_*_*', 
        'keep *_selectedPatTracks_*_*',
        'keep *_dimuons_*_*', 
        'keep *_dimuonsOneTrack_*_*', 
        'keep *_dimuonsGlobal_*_*', 
        'keep *_dimuonsOneStandAloneMuon_*_*', 
        'keep *_muonMatch_*_*', 
        'keep *_trackMuMatch_*_*', 
        'keep *_allDimuonsMCMatch_*_*',
        )
)
dimuonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'dimuonsPath',
           'dimuonsOneTrackPath')
    )
)

AODSIMDimuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMDimuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMDimuonEventContent.outputCommands.extend(dimuonsEventContent.outputCommands)

dimuonsOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMDimuonEventContent,
    dimuonsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dimuon'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('dimuons.root')
)

