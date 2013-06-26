import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMuJetsWp = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGCSVL_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGCSVL_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGCSVM_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGCSVM_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGCSVT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGCSVT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGCSVT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGJPL_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGJPL_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGJPM_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGJPM_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGJPT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGJPT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGJPT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTCHPT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTCHPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTCHPT_v9_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTCHPT_WP')
    ),
))
PoolDBESSourcebtagMuJetsWp.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
