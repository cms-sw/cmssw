import FWCore.ParameterSet.Config as cms

DigiToRawFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep FEDRawDataCollection_source_*_*', 
        'keep FEDRawDataCollection_rawDataCollector_*_*')
)

# foo bar baz
# sBRiql9PXlCVD
# P7a7YOgD96XMu
