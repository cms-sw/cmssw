import FWCore.ParameterSet.Config as cms

MuonShowerDigiFillerBlock = cms.PSet(
  ShowerDigiFillerParameters = cms.PSet(
      digiMaxDistanceX = cms.double(25.0),
      dtDigiCollectionLabel  = cms.InputTag("muonDTDigis"),
      cscDigiCollectionLabel = cms.InputTag("muonCSCDigis","MuonCSCStripDigi")
  )
)


