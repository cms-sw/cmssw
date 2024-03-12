import FWCore.ParameterSet.Config as cms

pfDQMEventSelector = cms.EDFilter("PFDQMEventSelector",
    DebugOn       = cms.bool(False),
    InputFileName = cms.string("Test.root"),
    FolderNames = cms.vstring("PFJet/CompWithGenJet","PFJet/CompWithCaloJet")
)
# foo bar baz
# 1R0rvQMaHlUu7
# A4rMzuJtcUIB3
