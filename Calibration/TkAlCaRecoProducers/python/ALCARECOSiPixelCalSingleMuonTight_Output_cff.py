import FWCore.ParameterSet.Config as cms

OutALCARECOSiPixelCalSingleMuonTight_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiPixelCalSingleMuonTight')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOSiPixelCalSingleMuonTight_*_*',
        'keep *_muons__*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_*riggerResults_*_HLT',
        'keep *_*closebyPixelClusters*_*_*',
        'keep *_*trackDistances*_*_*',
        'keep PileupSummaryInfos_addPileupInfo_*_*'
     )
)
OutALCARECOSiPixelCalSingleMuonTight=OutALCARECOSiPixelCalSingleMuonTight_noDrop.clone()
OutALCARECOSiPixelCalSingleMuonTight.outputCommands.insert(0, "drop *")
