import FWCore.ParameterSet.Config as cms

etaRangeCaloJetSelector = cms.EDFilter( "EtaRangeCaloJetSelector",
   src = cms.InputTag("hltCaloJetL1FastJetCorrected"),
   etaMin = cms.double( -99.9 ), #2.4
   etaMax = cms.double( +99.9 ), #2.4
)

