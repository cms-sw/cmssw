import FWCore.ParameterSet.Config as cms

l1uGTTestcrateTree = cms.EDAnalyzer( "L1uGTTreeProducer",
    ugtToken = cms.InputTag( "gtTestcrateStage2Digis" )
)
