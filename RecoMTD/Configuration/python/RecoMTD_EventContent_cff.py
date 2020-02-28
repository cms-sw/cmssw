import FWCore.ParameterSet.Config as cms

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*'
        )
)

#RECO content
RecoMTDRECO = RecoMTDAOD.copy()

#FEVT content
RecoMTDFEVT = RecoMTDRECO.copy()
