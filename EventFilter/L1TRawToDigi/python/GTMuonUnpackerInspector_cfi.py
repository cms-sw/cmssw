import FWCore.ParameterSet.Config as cms

GTMuonUnpackerInspector = cms.EDFilter(
    "GTMuonUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "Muon"),
    outputCollectionName = cms.String("inspectedGTMuons")
)