import FWCore.ParameterSet.Config as cms

#FEVT
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdTrackExtender_*_*'        
        )
)
#RECO content
RecoMTDRECO = RecoMTDFEVT.copy()
#AOD content
RecoMTDAOD = RecoMTDFEVT.copy()
