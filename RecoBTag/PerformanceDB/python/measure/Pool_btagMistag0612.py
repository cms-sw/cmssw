import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMistag0612 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVLtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVLwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGCSVTtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGCSVTwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGCSVTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPLtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPLwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPTtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPTwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPLtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPLwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPTtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPTwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGJPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHEMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHEMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHPTtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHPTwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHELtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHELwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHEMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHEMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPMtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPMwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPTtable_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPTwp_v8_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTwp_v8_offline')
    ),
))
PoolDBESSourcebtagMistag0612.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
