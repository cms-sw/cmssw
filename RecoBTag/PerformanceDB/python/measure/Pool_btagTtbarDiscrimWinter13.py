import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagTtbarDiscrim = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_TTBARDISCRIMBTAGCSV_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGCSV_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARDISCRIMBTAGCSV_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGCSV_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_TTBARDISCRIMBTAGJP_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGJP_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARDISCRIMBTAGJP_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGJP_WP')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('PerformancePayloadFromBinnedTFormula_TTBARDISCRIMBTAGTCHP_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGTCHP_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('PerformanceWorkingPoint_TTBARDISCRIMBTAGTCHP_v10_offline'),
    label = cms.untracked.string('TTBARDISCRIMBTAGTCHP_WP')
    ),
))
PoolDBESSourcebtagTtbarDiscrim.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'
