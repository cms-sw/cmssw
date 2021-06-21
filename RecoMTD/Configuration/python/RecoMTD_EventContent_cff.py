import FWCore.ParameterSet.Config as cms

#RECO content
RecoMTDRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)

#FEVT content
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMTDFEVT.outputCommands.extend(RecoMTDRECO.outputCommands)

#AOD
RecoMTDAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep intedmValueMap_trackExtenderWithMTD_*_*',
        'keep floatedmValueMap_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*')
)
