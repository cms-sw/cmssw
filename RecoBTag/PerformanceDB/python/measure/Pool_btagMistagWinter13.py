import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistagWinter13 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVL_v10_offline'),
    label = cms.untracked.string('MISTAGCSVL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVL_v10_offline'),
    label = cms.untracked.string('MISTAGCSVL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVM_v10_offline'),
    label = cms.untracked.string('MISTAGCSVM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVM_v10_offline'),
    label = cms.untracked.string('MISTAGCSVM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVT_v10_offline'),
    label = cms.untracked.string('MISTAGCSVT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVT_v10_offline'),
    label = cms.untracked.string('MISTAGCSVT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MISTAGCSVSLV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVV1L_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVV1L_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVV1M_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVV1M_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVV1T_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVV1T_v10_offline'),
    label = cms.untracked.string('MISTAGCSVV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPL_v10_offline'),
    label = cms.untracked.string('MISTAGJPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPL_v10_offline'),
    label = cms.untracked.string('MISTAGJPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPM_v10_offline'),
    label = cms.untracked.string('MISTAGJPM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPM_v10_offline'),
    label = cms.untracked.string('MISTAGJPM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPT_v10_offline'),
    label = cms.untracked.string('MISTAGJPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPT_v10_offline'),
    label = cms.untracked.string('MISTAGJPT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGTCHPT_v10_offline'),
    label = cms.untracked.string('MISTAGTCHPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGTCHPT_v10_offline'),
    label = cms.untracked.string('MISTAGTCHPT_WP')
    ),
))
PoolDBESSourcebtagMistagWinter13.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'
