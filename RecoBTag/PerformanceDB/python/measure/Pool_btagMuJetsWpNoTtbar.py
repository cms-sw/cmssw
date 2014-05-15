import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMuJetsWpNoTtbar = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1L_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVSLV1L_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1L_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1M_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVSLV1M_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1M_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1T_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARCSVSLV1T_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARCSVSLV1T_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARJPL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARJPL_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARJPM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARJPM_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPM_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARJPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARJPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARJPT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MUJETSWPBTAGNOTTBARTCHPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARTCHPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MUJETSWPBTAGNOTTBARTCHPT_v10_offline'),
    label = cms.untracked.string('MUJETSWPBTAGNOTTBARTCHPT_WP')
    ),
))
PoolDBESSourcebtagMuJetsWpNoTtbar.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'
