import FWCore.ParameterSet.Config as cms

def regressionWeights(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_EBCorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALonly'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_lowpt_EBCorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALonly_lowpt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALonly_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_lowpt_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALonly_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALonly'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_lowpt_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALonly_lowpt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALonly_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_lowpt_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALonly_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_EBCorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALTRK'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_lowpt_EBCorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALTRK_lowpt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALTRK_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_lowpt_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_eb_ECALTRK_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALTRK'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_lowpt_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALTRK_lowpt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALTRK_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),                                              
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDelectron_track_lowpt_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('electron_ee_ECALTRK_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),                                              
        cms.PSet(
           record = cms.string('GBRDWrapperRcd'),
           tag = cms.string('GEDphoton_EBCorrection_80X_EGM_v4'),
           label = cms.untracked.string('photon_eb_ECALonly'),
           connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
           record = cms.string('GBRDWrapperRcd'),
           tag = cms.string('GEDphoton_lowpt_EBCorrection_80X_EGM_v4'),
           label = cms.untracked.string('photon_eb_ECALonly_lowpt'),
           connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('photon_eb_ECALonly_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),            
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_lowpt_EBUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('photon_eb_ECALonly_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        ),            
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('photon_ee_ECALonly'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_lowpt_EECorrection_80X_EGM_v4'),
            label = cms.untracked.string('photon_ee_ECALonly_lowpt'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('photon_ee_ECALonly_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            ),
        cms.PSet(
            record = cms.string('GBRDWrapperRcd'),
            tag = cms.string('GEDphoton_lowpt_EEUncertainty_80X_EGM_v4'),
            label = cms.untracked.string('photon_ee_ECALonly_lowpt_var'),
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
            )
    ))
    return process


