import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVLbtable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVLbwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVLctable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVLcwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVLltable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVLlwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVLlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVMbtable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVMbwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVMctable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVMcwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVMltable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVMlwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVMlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVTbtable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVTbwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVTctable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVTcwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCSSVTltable_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCSSVTlwp_v1_offline'),
    label = cms.untracked.string('BTagMCSSVTlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHELbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHELbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHELctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHELcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHELltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHELlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHELlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHEMbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHEMbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHEMctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHEMcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHEMltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHEMlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHEMlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHETbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHETbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHETctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHETcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHETltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHETlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHETlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPLbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPLbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPLctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPLcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPLltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPLlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPLlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPMbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPMbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPMctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPMcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPMltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPMlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPMlwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPTbtable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTbtable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPTbwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTbwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPTctable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTctable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPTcwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTcwp_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCTCHPTltable_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTltable_v1_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCTCHPTlwp_v1_offline'),
    label = cms.untracked.string('BTagMCTCHPTlwp_v1_offline')
    ),
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

                              
                              
                              
