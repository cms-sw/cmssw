import FWCore.ParameterSet.Config as cms
from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import dtunpacker

muonDTDigis = dtunpacker.clone()

hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
 inputDCC         = cms.InputTag( "dttfDigis" ),   
 inputDDU         = cms.InputTag( "muonDTDigis" ),   
 inputDigis       = cms.InputTag( "muonDTDigis" ),   
 processDCC       = cms.bool( False ),   
 processDDU       = cms.bool( False ),   
 processDigis     = cms.bool( True ),   
 processingMode   = cms.int32( 0 ),   # 0=(DCC | DDU) | Digis/ 
                                      # 1=(DCC & DDU) | Digis/
                                      # 2=(DCC | DDU) & Digis/
                                      # 3=(DCC & DDU) & Digis/   
 minChamberLayers = cms.int32( 6 ),
 maxStation       = cms.int32( 3 ),
 minQual          = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH/
 minDDUBX         = cms.int32( 9 ),
 maxDDUBX         = cms.int32( 14 ),
 minActiveChambs  = cms.int32( 1 ),
 activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12)
)

# this is for filtering on HLT path
HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

dtHLTSkim = cms.Path(HLTDT)

dtSkim=cms.Path(muonDTDigis+hltDTActivityFilter)
