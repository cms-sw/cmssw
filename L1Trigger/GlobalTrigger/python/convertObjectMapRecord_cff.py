import FWCore.ParameterSet.Config as cms
from L1Trigger.GlobalTrigger.convertObjectMapRecord_cfi import hltL1GtObjectMap

simGtDigis = hltL1GtObjectMap.clone()
simGtDigis.L1GtObjectMapTag = cms.InputTag( "simGtDigis" )

convertObjectMapRecords = cms.Sequence(hltL1GtObjectMap*simGtDigis)
