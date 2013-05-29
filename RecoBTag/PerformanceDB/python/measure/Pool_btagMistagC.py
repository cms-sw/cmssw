import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistagC = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVLC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVLC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVMC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVMC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVTC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVTC_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPLC_v9_offline'),
    label = cms.untracked.string('MISTAGJPLC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPLC_v9_offline'),
    label = cms.untracked.string('MISTAGJPLC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPMC_v9_offline'),
    label = cms.untracked.string('MISTAGJPMC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPMC_v9_offline'),
    label = cms.untracked.string('MISTAGJPMC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPTC_v9_offline'),
    label = cms.untracked.string('MISTAGJPTC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPTC_v9_offline'),
    label = cms.untracked.string('MISTAGJPTC_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGTCHPTC_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTC_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGTCHPTC_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTC_WP')
    ),
))
PoolDBESSourcebtagMistagC.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
