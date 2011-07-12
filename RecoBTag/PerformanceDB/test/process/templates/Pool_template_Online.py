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
    tag = cms.string('BTagTEMPLATEtable_VERSION_offline'),
    label = cms.untracked.string('BTagTEMPLATEtable_VERSION_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagTEMPLATEwp_VERSION_offline'),
    label = cms.untracked.string('BTagTEMPLATEwp_VERSION_offline')
    ),
))
PoolDBESSource.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
                              
                              
                              
