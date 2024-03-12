import FWCore.ParameterSet.Config as cms

hltCandViewSelector = cms.EDFilter("CandViewSelector",
   src = cms.InputTag( "hltCollection" ),
   cut = cms.string( "pt()>-1" )
)
# foo bar baz
# TmQR0XcP5ZQSs
