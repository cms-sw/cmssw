import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtag = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGCSVLtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGCSVLwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGCSVMtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGCSVMwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGCSVTtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGCSVTwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGCSVTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJBPLtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJBPLwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJBPMtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJBPMwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJBPTtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJBPTwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJBPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJPLtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJPLwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJPMtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJPMwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGJPTtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGJPTwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGJPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGSSVHEMtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGSSVHEMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGSSVHEMwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGSSVHEMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGSSVHPTtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGSSVHPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGSSVHPTwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGSSVHPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGTCHELtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGTCHELtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGTCHELwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGTCHELwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGTCHEMtable_v7_offline'),
    label = cms.untracked.string('BTagBTAGTCHEMtable_v7_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGTCHEMwp_v7_offline'),
    label = cms.untracked.string('BTagBTAGTCHEMwp_v7_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGTCHPMtable_v7_offline'),
    label = cms.untracked.string('BTagBTAGTCHPMtable_v7_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGTCHPMwp_v7_offline'),
    label = cms.untracked.string('BTagBTAGTCHPMwp_v7_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagBTAGTCHPTtable_v6_offline'),
    label = cms.untracked.string('BTagBTAGTCHPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagBTAGTCHPTwp_v6_offline'),
    label = cms.untracked.string('BTagBTAGTCHPTwp_v6_offline')
    ),
))
PoolDBESSourcebtag.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'

                              
                              
                              
