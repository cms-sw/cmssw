import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
electronIdPdfs = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ElectronLikelihoodPdfsRcd'),
        tag = cms.string('ElectronLikelihoodPdfsRcd_tag')
    ))
)

CondDBCommon.connect = 'sqlite_file:/afs/cern.ch/user/e/emanuele/public/4Likelihood/PDFsSQLite/CMSSW_2_0_X/electronIdLikelihoodTkIsolated.db'

