import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
# for CMSSW >= 1_6_8
#replace CondDBCommon.connect = "sqlite_file:/afs/cern.ch/user/e/emanuele/public/4Likelihood/PDFsSQLite/CMSSW_1_6_8/electronIdLikelihoodTkIsolated.db"    
#replace CondDBCommon.catalog = "file:/afs/cern.ch/user/e/emanuele/public/4Likelihood/PDFsSQLite/CMSSW_1_6_8/electronIdLikelihoodTkIsolated.xml"
PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ElectronLikelihoodPdfsRcd'),
        tag = cms.string('ElectronLikelihoodPdfsRcd_tag')
    ))
)

# for CMSSW >= 1_7_X and for 2_0_0
CondDBCommon.connect = cms.InputTag("sqlite_fip","electronIdLikelihoodTkIsolated.db")

