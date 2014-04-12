import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourceCalo33 = cms.ESSource("PoolDBESSource",
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
))
PoolDBESSourceCalo33.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
