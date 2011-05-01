import FWCore.ParameterSet.Config as cms

hltElectronGenericFilter = cms.EDFilter("HLTElectronGenericFilter",
   saveTags = cms.bool( False ),

   lessThan = cms.bool(True),			  			  
   thrRegularEB = cms.double(0.0),
   thrRegularEE = cms.double(0.0),	  
   thrOverPtEB 	= cms.double(-1.0),	  
   thrOverPtEE   = cms.double(-1.0),		  
   thrTimesPtEB  = cms.double(-1.0),		  
   thrTimesPtEE  = cms.double(-1.0),		  
   									 
   isoTag =  cms.InputTag("hltSingleElectronTrackIsol"),
   nonIsoTag = cms.InputTag("hltSingleElectronHcalTrackIsol"),               
   doIsolated = cms.bool(True),
   ncandcut = cms.int32(1),
   candTag = cms.InputTag("hltSingleElectronOneOEMinusOneOPFilter"),

   L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
   L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso") 
)


