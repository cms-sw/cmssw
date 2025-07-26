import FWCore.ParameterSet.Config as cms

GTEGammaUnpackerInspector = cms.EDFilter(
    "GTEGammaUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "EGamma"),
    outputCollectionName = cms.String("inspectedGTEGammas")
)