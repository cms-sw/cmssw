import FWCore.ParameterSet.Config as cms

hltSelector4Jets = cms.EDFilter( "LargestEtPFJetSelector",
    src = cms.InputTag( "hltMCJetCorJetIcone5" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 4 )
)

