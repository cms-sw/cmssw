import FWCore.ParameterSet.Config as cms

hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
   inputDCC         = cms.InputTag( "hltDTTFUnpacker" ), 
   inputDDU         = cms.InputTag( "hltMuonDTDigis" ), 
   inputDigis       = cms.InputTag( "hltMuonDTDigis" ), 
   processDCC       = cms.bool( True ), 
   processDDU       = cms.bool( True ), 
   processDigis     = cms.bool( True ), 
   processingMode   = cms.int32( 2 ),   # 0=(DCC | DDU) | Digis
                                        # 1=(DCC & DDU) | Digis
                                        # 2=(DCC | DDU) & Digis
                                        # 3=(DCC & DDU) & Digis
   minChamberLayers = cms.int32( 6 ),
   maxStation       = cms.int32( 3 ),
   minQual          = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH
   minDDUBX         = cms.int32( 9 ),
   maxDDUBX         = cms.int32( 14 ),
   minActiveChambs  = cms.int32( 1 ),
   activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12)
                                    )
