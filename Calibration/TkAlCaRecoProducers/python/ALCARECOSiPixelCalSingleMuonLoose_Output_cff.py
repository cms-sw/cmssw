import FWCore.ParameterSet.Config as cms

OutALCARECOSiPixelCalSingleMuonLoose_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiPixelCalSingleMuonLoose')
    ),
    outputCommands = cms.untracked.vstring(
      'keep *_ALCARECOSiPixelCalSingleMuonLoose_*_*',
      'keep *_muons__*',
      'keep *_offlinePrimaryVertices_*_*',
      'keep *_*riggerResults_*_HLT',
      'keep PixelFEDChanneledmNewDetSetVector_siPixelDigis_*_*'
     )
)

OutALCARECOSiPixelCalSingleMuonLoose=OutALCARECOSiPixelCalSingleMuonLoose_noDrop.clone()
OutALCARECOSiPixelCalSingleMuonLoose.outputCommands.insert(0, "drop *")
