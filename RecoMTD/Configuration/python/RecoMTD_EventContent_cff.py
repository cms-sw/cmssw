import FWCore.ParameterSet.Config as cms

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep intedmValueMap_trackExtenderWithMTD_*_*',
        'keep floatedmValueMap_trackExtenderWithMTD_*_*',
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
