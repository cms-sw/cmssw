import FWCore.ParameterSet.Config as cms
from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import dtunpacker

muonDTDigis = dtunpacker.clone()

import HLTrigger.special.hltDTActivityFilter_cfi

hltDTActivityFilter = HLTrigger.special.hltDTActivityFilter_cfi.hltDTActivityFilter.clone(
    inputDCC         = cms.InputTag( "dttfDigis" ),   
    inputDDU         = cms.InputTag( "muonDTDigis" ),   
    inputDigis       = cms.InputTag( "muonDTDigis" ),   
    processDCC       = cms.bool( False ),   
    processDDU       = cms.bool( False ),
    processRPC       = cms.bool( False ),
    processDigis     = cms.bool( True ),   
    minChamberLayers = cms.int32( 6 ),
    minDDUBX         = cms.int32( 9 ),
    maxDDUBX         = cms.int32( 14 ),
)

# this is for filtering on HLT path
HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

dtHLTSkimseq = cms.Sequence(HLTDT)

dtSkimseq=cms.Sequence(muonDTDigis+hltDTActivityFilter)
