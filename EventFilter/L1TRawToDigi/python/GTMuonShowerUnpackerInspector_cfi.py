import FWCore.ParameterSet.Config as cms

GTMuonShowerUnpackerInspector = cms.EDFilter(
    "GTMuonShowerUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "MuonShower"),
    outputCollectionName = cms.String("inspectedGTMuonShowers")
)