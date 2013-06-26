import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcebtagMuJetsWp0612 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVLtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVLwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVTtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGCSVTwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGCSVTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPLtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPLwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPTtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJPTwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPLtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPLtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPLwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPLwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPTtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGJBPTwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGJBPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGSSVHEMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGSSVHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGSSVHEMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGSSVHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGSSVHPTtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGSSVHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGSSVHPTwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGSSVHPTwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHELtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHELtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHELwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHELwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHEMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHEMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHEMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHEMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHPMtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHPMtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHPMwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHPMwp_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHPTtable_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHPTtable_v8_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMUJETSWPBTAGTCHPTwp_v8_offline'),
    label = cms.untracked.string('BTagMUJETSWPBTAGTCHPTwp_v8_offline')
    ),
))
PoolDBESSourcebtagMuJetsWp0612.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

                              
                              
                              
