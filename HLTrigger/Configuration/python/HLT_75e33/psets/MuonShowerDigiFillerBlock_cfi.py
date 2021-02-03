import FWCore.ParameterSet.Config as cms

MuonShowerDigiFillerBlock = cms.PSet(
    ShowerDigiFillerParameters = cms.PSet(
        cscDigiCollectionLabel = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
        digiMaxDistanceX = cms.double(25.0),
        dtDigiCollectionLabel = cms.InputTag("muonDTDigis")
    )
)