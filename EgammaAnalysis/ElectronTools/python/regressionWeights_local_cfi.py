from CondCore.DBCommon.CondDBSetup_cfi import *

GBRDWrapperRcd  =  cms.ESSource("PoolDBESSource",
                                CondDBSetup,
                                DumpStat=cms.untracked.bool(True),
                                timetype = cms.string('runnumber'),
                                connect = cms.string('sqlite_file:/afs/cern.ch/work/r/rcoelhol/RegressionPR/80release/newvars/database/RegressionDatabase/SQLiteFiles/GED_80X_Winter2016/ged_regression_20170114.db'),
                                toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALonly_lowpt'),
            label = cms.untracked.string('electron_eb_ECALonly_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALonly'),
            label = cms.untracked.string('electron_eb_ECALonly')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALonly_lowpt_var'),
            label = cms.untracked.string('electron_eb_ECALonly_lowpt_var')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALonly_var'),
            label = cms.untracked.string('electron_eb_ECALonly_var')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALonly_lowpt'),
            label = cms.untracked.string('electron_ee_ECALonly_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALonly'),
            label = cms.untracked.string('electron_ee_ECALonly')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALonly_lowpt_var'),
            label = cms.untracked.string('electron_ee_ECALonly_lowpt_var')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALonly_var'),
            label = cms.untracked.string('electron_ee_ECALonly_var')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALTRK_lowpt'),
            label = cms.untracked.string('electron_eb_ECALTRK_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALTRK'),
            label = cms.untracked.string('electron_eb_ECALTRK')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALTRK_lowpt_var'),
            label = cms.untracked.string('electron_eb_ECALTRK_lowpt_var')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_eb_ECALTRK_var'),
            label = cms.untracked.string('electron_eb_ECALTRK_var')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALTRK_lowpt'),
            label = cms.untracked.string('electron_ee_ECALTRK_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALTRK'),
            label = cms.untracked.string('electron_ee_ECALTRK')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALTRK_lowpt_var'),
            label = cms.untracked.string('electron_ee_ECALTRK_lowpt_var')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('electron_ee_ECALTRK_var'),
            label = cms.untracked.string('electron_ee_ECALTRK_var')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_eb_ECALonly_lowpt'),
            label = cms.untracked.string('photon_eb_ECALonly_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_eb_ECALonly'),
            label = cms.untracked.string('photon_eb_ECALonly')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_eb_ECALonly_lowpt_var'),
            label = cms.untracked.string('photon_eb_ECALonly_lowpt_var')
            ),            
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_eb_ECALonly_var'),
            label = cms.untracked.string('photon_eb_ECALonly_var')
            ),            
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_ee_ECALonly_lowpt'),
            label = cms.untracked.string('photon_ee_ECALonly_lowpt')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_ee_ECALonly'),
            label = cms.untracked.string('photon_ee_ECALonly')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_ee_ECALonly_lowpt_var'),
            label = cms.untracked.string('photon_ee_ECALonly_lowpt_var')
            ),    
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('photon_ee_ECALonly_var'),
            label = cms.untracked.string('photon_ee_ECALonly_var')
            )),    
)

