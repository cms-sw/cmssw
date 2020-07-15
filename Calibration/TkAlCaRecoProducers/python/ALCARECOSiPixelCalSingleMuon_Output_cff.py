import FWCore.ParameterSet.Config as cms

OutALCARECOSiPixelCalSingleMuon_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiPixelCalSingleMuon')
    ),
    outputCommands = cms.untracked.vstring(
      'keep *_ALCARECOSiPixelCalSingleMuon_*_*',
      'keep *_muons__*',
      'keep *_offlinePrimaryVertices_*_*',
      'keep *_*riggerResults_*_HLT'
     )
)
import copy

OutALCARECOSiPixelCalSingleMuon=copy.deepcopy(OutALCARECOSiPixelCalSingleMuon_noDrop)
OutALCARECOSiPixelCalSingleMuon.outputCommands.insert(0, "drop *")