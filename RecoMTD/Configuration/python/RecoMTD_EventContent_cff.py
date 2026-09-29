import FWCore.ParameterSet.Config as cms

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *edmValueMap_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)

#RECO content
RecoMTDRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
         'keep recoTrack*_trackExtenderWithMTD_*_*',
         'keep TrackingRecHitsOwned_trackExtenderWithMTD_*_*',
    )
)
RecoMTDRECO.outputCommands.extend(RecoMTDAOD.outputCommands)

#FEVT content
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMTDFEVT.outputCommands.extend(RecoMTDRECO.outputCommands)

#FEVTHLT content
RecoMTDFEVTHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *edmValueMap_hltTrackExtenderWithMTD_*_*',
        'keep *_hltMtdTrackQualityMVA_*_*',
        'keep recoTrack*_hltTrackExtenderWithMTD_*_*',
        'keep TrackingRecHitsOwned_hltTrackExtenderWithMTD_*_*',
    )
)
