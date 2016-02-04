import FWCore.ParameterSet.Config as cms

hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
   inputDCC         = cms.InputTag( "hltDTTFUnpacker" ), 
   inputDDU         = cms.InputTag( "hltMuonDTDigis" ), 
   inputRPC         = cms.InputTag( "hltGtDigis" ), 
   inputDigis       = cms.InputTag( "hltMuonDTDigis" ), 
   processDCC       = cms.bool( True ), 
   processDDU       = cms.bool( True ), 
   processRPC       = cms.bool( True ), 
   processDigis     = cms.bool( True ), 

   maxDeltaPhi = cms.double( 1.0 ),
   maxDeltaEta = cms.double( 0.3 ),

   orTPG         = cms.bool( True ),
   orRPC         = cms.bool( True ),
   orDigi        = cms.bool( False ), # && of trig & digi info
   
   minChamberLayers = cms.int32( 5 ),
   maxStation       = cms.int32( 3 ),
   minTPGQual       = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH
   minDDUBX         = cms.int32( 8 ),
   maxDDUBX         = cms.int32( 13 ),
   minDCCBX         = cms.int32( -1 ),
   maxDCCBX         = cms.int32( 1 ),
   minRPCBX         = cms.int32( -1 ),
   maxRPCBX         = cms.int32( 1 ),
   minActiveChambs  = cms.int32( 1 ),
   activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12)
                                    )
