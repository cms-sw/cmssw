import FWCore.ParameterSet.Config as cms

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)

#RECO content
RecoMTDRECO = cms.PSet(
    outputCommands = cms.untracked.vstring() 
)
RecoMTDRECO.outputCommands.extend(RecoMTDAOD.outputCommands)

#FEVT content
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring() 
)
RecoMTDFEVT.outputCommands.extend(RecoMTDRECO.outputCommands)
