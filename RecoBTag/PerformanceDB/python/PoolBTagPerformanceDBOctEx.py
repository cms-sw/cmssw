import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourceOctEx = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8SSVMtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8SSVMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8SSVMwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8SSVMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8SSVTtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8SSVTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8SSVTwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8SSVTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHELtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHELtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHELwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHELwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHEMtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHEMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHEMwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHEMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHETtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHETtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHETwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHETwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHPLtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHPLwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHPMtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHPMwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagSYSTEM8TCHPTtable_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagSYSTEM8TCHPTwp_v1_offline'),
    label = cms.untracked.string('BTagSYSTEM8TCHPTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPLtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPLwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPMtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPMwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPTtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPTwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGJPTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVMtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGSSVMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVMwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGSSVMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHELtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHELwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHEMtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHEMwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPMtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPMwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPTtable_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPTwp_v1_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJBPLtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJBPLwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJBPMtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJBPMwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJBPTtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJBPTwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJBPTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJPLtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJPLwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJPMtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJPMwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELJPTtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELJPTwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELJPTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELSSVLtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELSSVLwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELSSVMtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELSSVMwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELSSVTtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELSSVTwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELSSVTwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHELtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHELtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHELwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHELwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHEMtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHEMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHEMwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHEMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHETtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHETtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHETwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHETwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHPLtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPLtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHPLwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPLwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHPMtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPMtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHPMwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPMwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagPTRELTCHPTtable_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPTtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagPTRELTCHPTwp_v1_offline'),
    label = cms.untracked.string('BTagPTRELTCHPTwp_v1_offline')
    ),
))
PoolDBESSourceOctEx.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
