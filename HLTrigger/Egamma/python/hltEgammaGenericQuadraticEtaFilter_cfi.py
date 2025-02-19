import FWCore.ParameterSet.Config as cms

hltEgammaGenericQuadraticEtaFilter = cms.EDFilter("HLTEgammaGenericQuadraticEtaFilter",
   saveTags = cms.bool( False ),

   lessThan = cms.bool(True),			  
   useEt  = cms.bool(True),			  
   etaBoundaryEB12 = cms.double( 1.0 ),
   etaBoundaryEE12 = cms.double( 2.0 ),
   thrRegularEB1 = cms.double(4.0),
   thrRegularEB2 = cms.double(6.0),
   thrRegularEE1 = cms.double(6.0),	  
   thrRegularEE2 = cms.double(4.0),	  
   thrOverEEB1 	= cms.double(0.0020),	  
   thrOverEEB2 	= cms.double(0.0020),	  
   thrOverEEE1   = cms.double(0.0020),		  
   thrOverEEE2   = cms.double(0.0020),		  
   thrOverE2EB1  = cms.double(0.0),		  
   thrOverE2EB2  = cms.double(0.0),		  
   thrOverE2EE1  = cms.double(0.0),		  
   thrOverE2EE2  = cms.double(0.0),		  
   									 
   isoTag =  cms.InputTag("hltEGIsol"),
   nonIsoTag = cms.InputTag("hltEGNonIsol"),               
   doIsolated = cms.bool(False),
   ncandcut = cms.int32(1),
   candTag = cms.InputTag("hltEGIsolFilter"),

   L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
   L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate") 

)


