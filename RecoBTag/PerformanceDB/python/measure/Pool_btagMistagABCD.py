import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistagABCD = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVLABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVLABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVLABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVMABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVMABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVMABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGCSVTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGCSVTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGCSVTABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPLABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPLABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPLABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPLABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPMABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPMABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPMABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPMABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGJPTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPTABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGJPTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGJPTABCD_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_MISTAGTCHPTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTABCD_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_MISTAGTCHPTABCD_v9_offline'),
    label = cms.untracked.string('MISTAGTCHPTABCD_WP')
    ),
))
PoolDBESSourcebtagMistagABCD.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
