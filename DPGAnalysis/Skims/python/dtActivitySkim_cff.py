import FWCore.ParameterSet.Config as cms
from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import dtunpacker

muonDTDigisforDTActivitySkim = dtunpacker.clone()

import HLTrigger.special.hltDTActivityFilter_cfi

hltDTActivityFilter = HLTrigger.special.hltDTActivityFilter_cfi.hltDTActivityFilter.clone(
   inputDCC         = cms.InputTag( "dttfDigis" ),
    inputDDU         = cms.InputTag( "muonDTDigisforDTActivitySkim" ),
    inputRPC         = cms.InputTag( "hltGtDigis" ),
    inputDigis       = cms.InputTag( "muonDTDigis" ),
    processDCC       = cms.bool( False ),
    processDDU       = cms.bool( False ),
    processRPC       = cms.bool( False ),
    processDigis     = cms.bool( True ),

    maxDeltaPhi = cms.double( 1.0 ),
    maxDeltaEta = cms.double( 0.3 ),

    orTPG         = cms.bool( True ),
    orRPC         = cms.bool( True ),
    orDigi        = cms.bool( True ),

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

# this is for filtering on HLT path
HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT','HLT_Activity_DT_Tuned'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

dtHLTSkimseq = cms.Sequence(HLTDT)

dtSkimseq=cms.Sequence(muonDTDigisforDTActivitySkim+hltDTActivityFilter)
