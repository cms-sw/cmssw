import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagTtbarMc0612 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGCSVtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGCSVtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGCSVwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGCSVwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGJPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGJPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGJPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGJPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGJBPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGJBPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGJBPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGJBPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGSSVHEtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGSSVHEtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGSSVHEwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGSSVHEwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGSSVHPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGSSVHPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGSSVHPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGSSVHPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGTCHEtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGTCHEtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGTCHEwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGTCHEwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARMCBTAGTCHPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGTCHPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARMCBTAGTCHPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARMCBTAGTCHPwp_v8_offline')
    ),
))
PoolDBESSourcebtagTtbarMc0612.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
