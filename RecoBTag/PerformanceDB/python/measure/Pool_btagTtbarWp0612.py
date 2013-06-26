import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagTtbarWp0612 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVLtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVLwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGCSVTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGCSVTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPLtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPLwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJPTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPLtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPLwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPLtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPLwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGJBPTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGJBPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHEMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHEMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHETtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHETtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHETwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHETwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHPTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGSSVHPTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGSSVHPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHELtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHELtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHELwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHELwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHEMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHEMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHETtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHETtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHETwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHETwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPLtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPLwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPMtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPMwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPTtable_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTTBARWPBTAGTCHPTwp_v8_offline'),
    label = cms.untracked.string('BTagTTBARWPBTAGTCHPTwp_v8_offline')
    ),
))
PoolDBESSourcebtagTtbarWp0612.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
