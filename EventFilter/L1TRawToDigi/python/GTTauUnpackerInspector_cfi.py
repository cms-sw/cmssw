import FWCore.ParameterSet.Config as cms

GTTauUnpackerInspector = cms.EDFilter(
    "GTTauUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "Tau"),
    outputCollectionName = cms.String("inspectedGTTaus")
)