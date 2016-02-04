import FWCore.ParameterSet.Config as cms

rctTextToRctDigi = cms.EDProducer("RctTextToRctDigi",
    FileEventOffset = cms.int32(0),
    TextFileName = cms.string('data/testElectronsRct_'),
    RctOutputLabel = cms.InputTag("RCTRegionSumsEmCands")
)


