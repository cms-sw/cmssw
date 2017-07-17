import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMuJetsWpTtbar = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARCSVSLV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGTTBARTCHPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARTCHPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGTTBARTCHPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGTTBARTCHPT_WP')
    ),
))
PoolDBESSourcebtagMuJetsWpTtbar.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'
