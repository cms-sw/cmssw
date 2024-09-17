import FWCore.ParameterSet.Config as cms

hltCandPtrSelector = cms.EDFilter("CandPtrSelector",
   src = cms.InputTag( "hltCollection" ),
   cut = cms.string( "pt()>-1" )
)
