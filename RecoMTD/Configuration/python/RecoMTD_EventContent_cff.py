import FWCore.ParameterSet.Config as cms

#FEVT content
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)

#RECO content
RecoMTDRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMTDRECO.outputCommands.extend(RecoMTDFEVT.outputCommands)

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep intedmValueMap_trackExtenderWithMTD_*_*',
        'keep floatedmValueMap_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)
