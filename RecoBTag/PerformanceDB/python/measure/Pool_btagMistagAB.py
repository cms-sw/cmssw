import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistagAB = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVLAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVLAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVMAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVMAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVTAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVTAB_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPLAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPLAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPLAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPLAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPMAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPMAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPMAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPMAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPTAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPTAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPTAB_v9_offline'),
    label = cms.untracked.string('MISTAGJPTAB_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGTCHPTAB_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTAB_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGTCHPTAB_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTAB_WP')
    ),
))
PoolDBESSourcebtagMistagAB.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
