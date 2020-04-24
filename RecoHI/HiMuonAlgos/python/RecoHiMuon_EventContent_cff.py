import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoHiMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#RECO content
RecoHiMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#AOD content
RecoHiMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#Add Isolation
from RecoMuon.MuonIsolationProducers.muIsolation_EventContent_cff import *
# AOD content for re-muons
reRecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_remuons_*_*',
                                           'keep *_*_remuons_*',
                                           # Tracks known by the Muon obj
                                           'keep recoTracks_standAloneMuons_*_*', 
                                           'keep recoTrackExtras_standAloneMuons_*_*', 
                                           'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
                                           'keep recoTracks_reglobalMuons_*_*', 
                                           'keep recoTrackExtras_reglobalMuons_*_*', 
                                           'keep recoTracks_retevMuons_*_*', 
                                           'keep recoTrackExtras_retevMuons_*_*', 
                                           'keep recoTracksToOnerecoTracksAssociation_retevMuons_*_*'
                                           )
)
# RECO content
reRecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MuonSeed_*_*',
                                           'keep *_ancientMuonSeed_*_*',
                                           'keep *_mergedStandAloneMuonSeeds_*_*',
                                           'keep TrackingRecHitsOwned_reglobalMuons_*_*', 
                                           'keep TrackingRecHitsOwned_retevMuons_*_*',
                                           'keep recoCaloMuons_recalomuons_*_*')
)
# Full Event content 
reRecoMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
reRecoMuonRECO.outputCommands.extend(reRecoMuonAOD.outputCommands)
reRecoMuonFEVT.outputCommands.extend(reRecoMuonRECO.outputCommands)

RecoHiMuonAOD.outputCommands.extend(reRecoMuonAOD.outputCommands)
RecoHiMuonRECO.outputCommands.extend(reRecoMuonRECO.outputCommands)
RecoHiMuonFEVT.outputCommands.extend(reRecoMuonFEVT.outputCommands)

