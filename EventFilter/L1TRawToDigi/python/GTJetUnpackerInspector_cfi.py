import FWCore.ParameterSet.Config as cms

GTJetUnpackerInspector = cms.EDFilter(
    "GTJetUnpackerInspector",
    objectSrc = cms.InputTag("gtStage2Digis", "Jet"),
    outputCollectionName = cms.String("inspectedGTJets")
)