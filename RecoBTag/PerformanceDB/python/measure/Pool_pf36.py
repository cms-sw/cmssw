import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourcePf36 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHELbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHELbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHELctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHELcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHELltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHELlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHELlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHEMbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHEMbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHEMctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHEMcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHEMltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHEMlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHEMlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHETbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHETbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHETctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHETcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHETltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHETlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHETlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHPTbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHPTbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHPTctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHPTcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfSSVHPTltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfSSVHPTlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfSSVHPTlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHELbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHELbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHELctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHELcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHELltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHELlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHELlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHEMbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHEMbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHEMctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHEMcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHEMltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHEMlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHEMlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHETbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHETbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHETctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHETcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHETltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHETlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHETlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPLbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPLbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPLctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPLcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPLltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPLlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPLlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPMbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPMbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPMctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPMcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPMltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPMlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPMlwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPTbtable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTbtable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPTbwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTbwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPTctable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTctable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPTcwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTcwp_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCPfTCHPTltable_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTltable_pf_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCPfTCHPTlwp_pf_v2_offline'),
    label = cms.untracked.string('BTagMCPfTCHPTlwp_pf_v2_offline')
    ),
))
PoolDBESSourcePf36.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
