import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
electronIdPdfs = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ElectronLikelihoodPdfsRcd'),
        tag = cms.string('ElectronLikelihoodPdfs_v3_offline')
    ))
)

electronIdPdfs.connect = 'frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'


