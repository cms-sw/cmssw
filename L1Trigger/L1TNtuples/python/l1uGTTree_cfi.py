import FWCore.ParameterSet.Config as cms

l1uGTTree = cms.EDAnalyzer( "L1uGTTreeProducer",
    ugtToken = cms.InputTag( "ugtStage2Digis" )
)
