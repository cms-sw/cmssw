# /dev/CMSSW_11_1_0/PRef

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3', 
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v5', 
    'HLT_ZeroBias_v6'
)


# stream PhysicsEndOfFill

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetEmptyBX_selector
streamPhysicsEndOfFill_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_HIL1NotBptxORForPPRef_v2', 
    'HLT_HIL1UnpairedBunchBptxMinusForPPRef_v2', 
    'HLT_HIL1UnpairedBunchBptxPlusForPPRef_v2'
)


# stream PhysicsHIZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias1_datasetHIZeroBias1_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part0_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias1_datasetHIZeroBias2_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part1_v6')


# stream PhysicsHIZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias2_datasetHIZeroBias3_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part2_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias2_datasetHIZeroBias4_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part3_v6')


# stream PhysicsHIZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias3_datasetHIZeroBias5_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part4_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias3_datasetHIZeroBias6_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part5_v6')


# stream PhysicsHIZeroBias4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias4_datasetHIZeroBias7_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part6_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias4_datasetHIZeroBias8_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part7_v6')


# stream PhysicsHIZeroBias5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias5_datasetHIZeroBias10_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part9_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias5_datasetHIZeroBias9_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part8_v6')


# stream PhysicsHIZeroBias6

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias6_datasetHIZeroBias11_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part10_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.l1tResults = cms.InputTag('')
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.throw      = cms.bool(False)
streamPhysicsHIZeroBias6_datasetHIZeroBias12_selector.triggerConditions = cms.vstring('HLT_HIZeroBias_part11_v6')

