import FWCore.ParameterSet.Config as cms

sourceCardTextToRctDigi = cms.EDFilter("SourceCardTextToRctDigi",
    fileEventOffset = cms.int32(0),
    TextFileName = cms.string('RctDigiToSourceCardText.dat')
)


