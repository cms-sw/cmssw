import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcemistag = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVLtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVLwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVTtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVTwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGCSVTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPLtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPLwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPTtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPTwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPLtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPLtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPLwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPLwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPTtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPTwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGJPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHEMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHEMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHPTtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHPTwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHELtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHELwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHEMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHEMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPMtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPMwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMwp_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPTtable_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTtable_v6_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPTwp_v6_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTwp_v6_offline')
    ),
))
PoolDBESSourcemistag.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'

                              
                              
                              
