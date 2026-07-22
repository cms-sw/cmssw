import FWCore.ParameterSet.Config as cms

GTEtSumUnpackerInspector = cms.EDFilter(
    "GTEtSumUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "EtSum"),
    outputCollectionName = cms.String("inspectedGTEtSums")
)