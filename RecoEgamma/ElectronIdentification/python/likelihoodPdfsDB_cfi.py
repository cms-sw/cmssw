import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
electronIdPdfs = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ElectronLikelihoodPdfsRcd'),
        tag = cms.string('ElectronLikelihoodPdfsRcd_tag')
    ))
)

electronIdPdfs.connect = 'sqlite_fip:CondCore/SQLiteData/data/electronIdLikelihoodTkIsolated.db'


