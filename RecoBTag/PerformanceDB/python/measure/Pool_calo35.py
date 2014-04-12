import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourceCalo35 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,                            
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHELbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHELbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHELctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHELcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHELltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHELlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHELlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHEMbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHEMbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHEMctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHEMcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHEMltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHEMlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHEMlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHETbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHETbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHETctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHETcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHETltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHETlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHETlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHPTbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHPTbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHPTctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHPTcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloSSVHPTltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloSSVHPTlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloSSVHPTlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHELbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHELbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHELctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHELcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHELltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHELlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHELlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHEMbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHEMbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHEMctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHEMcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHEMltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHEMlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHEMlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHETbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHETbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHETctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHETcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHETltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHETlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHETlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPLbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPLbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPLctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPLcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPLltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPLlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPLlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPMbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPMbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPMctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPMcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPMltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPMlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPMlwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPTbtable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTbtable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPTbwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTbwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPTctable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTctable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPTcwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTcwp_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMCCaloTCHPTltable_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTltable_calo_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMCCaloTCHPTlwp_calo_v2_offline'),
    label = cms.untracked.string('BTagMCCaloTCHPTlwp_calo_v2_offline')
    ),
))
PoolDBESSourceCalo35.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
