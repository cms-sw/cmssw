import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistagD = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVLD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVLD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVMD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVMD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVTD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVTD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPLD_v9_offline'),
    label = cms.untracked.string('MISTAGJPLD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPLD_v9_offline'),
    label = cms.untracked.string('MISTAGJPLD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPMD_v9_offline'),
    label = cms.untracked.string('MISTAGJPMD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPMD_v9_offline'),
    label = cms.untracked.string('MISTAGJPMD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPTD_v9_offline'),
    label = cms.untracked.string('MISTAGJPTD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPTD_v9_offline'),
    label = cms.untracked.string('MISTAGJPTD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGTCHPTD_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGTCHPTD_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTD_WP')
    ),
))
PoolDBESSourcebtagMistagD.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
