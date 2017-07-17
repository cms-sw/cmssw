import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagTtbarWp = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGCSVL_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGCSVL_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGCSVM_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGCSVM_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGCSVT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGCSVT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGCSVT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGJPL_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGJPL_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGJPM_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGJPM_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGJPT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGJPT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGJPT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromTable_TTBARWPBTAGTCHPT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGTCHPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARWPBTAGTCHPT_v10_offline'),
    label = cms.untracked.string('TTBARWPBTAGTCHPT_WP')
    ),
))
PoolDBESSourcebtagTtbarWp.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'
