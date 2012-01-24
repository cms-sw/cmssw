import FWCore.ParameterSet.Config as cms

hltDynamicPrescaler = cms.EDFilter( "HLTDynamicPrescaler",
   saveTags = cms.bool( False )
)
