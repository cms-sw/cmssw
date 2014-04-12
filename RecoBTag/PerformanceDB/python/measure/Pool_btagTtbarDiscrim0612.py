import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagTtbarDiscrim0612 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGCSVtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGCSVtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGCSVwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGCSVwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGJPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGJPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGJPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGJPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGJBPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGJBPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGJBPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGJBPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGSSVHEtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGSSVHEtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGSSVHEwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGSSVHEwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGSSVHPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGSSVHPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGSSVHPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGSSVHPwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGTCHEtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGTCHEtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGTCHEwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGTCHEwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGTCHPtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGTCHPtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARDISCRIMBTAGTCHPwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARDISCRIMBTAGTCHPwp_v8_offline')
    ),
))
PoolDBESSourcebtagTtbarDiscrim0612.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
