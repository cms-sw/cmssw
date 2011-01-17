import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

zMuMuEventContent =  cms.PSet(
    outputCommands = cms.untracked.vstring(
    ####   to access the position at the momentum system for global and standalone muons
    'keep recoTrackExtras_standAloneMuons_*_*',
    'keep recoTracks_standAloneMuons_UpdatedAtVtx_*',
    #'keep recoCaloMuons_calomuons_*_*',
    #'keep *_selectedPatMuonsTriggerMatch_*_*',
    #'keep *_selectedPatTracks_*_*',
    'keep *_userDataMuons_*_*',
    'keep *_userDataTracks_*_*',
    'keep *_userDataDimuons_*_*',
    'keep *_userDataDimuonsOneTrack_*_*',
    #'keep *_dimuons_*_*',
    #'keep *_dimuonsOneTrack_*_*',
    'keep *_dimuonsGlobal_*_*',
    'keep *_dimuonsOneStandAloneMuon_*_*',
    'keep *_dimuonsOneTrackerMuon_*_*',
    ### to access vertex information
    'keep *_offlineBeamSpot_*_*',
    'keep *_offlinePrimaryVertices_*_*',
    #'keep *_offlinePrimaryVerticesWithBS_*_*',
    ### to save jet information
    #'keep *_sisCone5CaloJets_*_*',
    #'keep *_ak5CaloJets_*_*',
    'keep *_muons_*_*',
    'keep *_pfMet_*_*',
    'keep *_ak5PFJets_*_*'
    
    )
)

zMuMuSubskimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

# to add the AOD output uncomment the following
#zMuMuSubskimEventContent.outputCommands.extend(AODEventContent.outputCommands)

zMuMuSubskimEventContent.outputCommands.extend(zMuMuEventContent.outputCommands)

zMuMuSubskimOutputModule = cms.OutputModule("PoolOutputModule",
    zMuMuSubskimEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'dimuonsPath',
           'dimuonsOneTrackPath')
    ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zmumu'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('zMuMuSubskim.root')
)

