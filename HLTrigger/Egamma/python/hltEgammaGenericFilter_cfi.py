import FWCore.ParameterSet.Config as cms

hltEgammaGenericFilter = cms.EDFilter("HLTEgammaGenericFilter",
   saveTags = cms.bool( False ),

   lessThan = cms.bool(True),			  
   useEt  = cms.bool(False),			  
   thrRegularEB = cms.double(0.0),
   thrRegularEE = cms.double(0.0),	  
   thrOverEEB 	= cms.double(-1.0),	  
   thrOverEEE   = cms.double(-1.0),		  
   thrOverE2EB  = cms.double(-1.0),		  
   thrOverE2EE  = cms.double(-1.0),		  
   									 
   isoTag =  cms.InputTag("hltSingleEgammaHcalIsol"),
   nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),               
   doIsolated = cms.bool(True),
   ncandcut = cms.int32(1),
   candTag = cms.InputTag("hltSingleEgammaEtFilter"),

   L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
   L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate") 
)


