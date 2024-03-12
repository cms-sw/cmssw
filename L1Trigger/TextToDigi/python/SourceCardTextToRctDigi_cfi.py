import FWCore.ParameterSet.Config as cms

sourceCardTextToRctDigi = cms.EDProducer("SourceCardTextToRctDigi",
    fileEventOffset = cms.int32(0),
    TextFileName = cms.string('RctDigiToSourceCardText.dat')
)


# foo bar baz
# 69mtAbH0MvMmw
# zH2i2qBXQqt30
