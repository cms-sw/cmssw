import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDBUser'),  

              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              
              electron_config = cms.PSet( # EB, EE
                regressionKey_ecalonly  = cms.vstring('electron_eb_ECALonly_lowpt', 'electron_eb_ECALonly', 'electron_ee_ECALonly_lowpt', 'electron_ee_ECALonly'),
                uncertaintyKey_ecalonly = cms.vstring('electron_eb_ECALonly_lowpt_var', 'electron_eb_ECALonly_var', 'electron_ee_ECALonly_lowpt_var', 'electron_ee_ECALonly_var'),
                regressionKey_ecaltrk  = cms.vstring('electron_eb_ECALTRK_lowpt', 'electron_eb_ECALTRK', 'electron_ee_ECALTRK_lowpt', 'electron_ee_ECALTRK'),
                uncertaintyKey_ecaltrk = cms.vstring('electron_eb_ECALTRK_lowpt_var', 'electron_eb_ECALTRK_var', 'electron_ee_ECALTRK_lowpt_var', 'electron_ee_ECALTRK_var'),
                                          ),
              
              photon_config   = cms.PSet( # EB, EE
                regressionKey_ecalonly  = cms.vstring('photon_eb_ECALonly_lowpt', 'photon_eb_ECALonly', 'photon_ee_ECALonly_lowpt', 'photon_ee_ECALonly'),
                uncertaintyKey_ecalonly = cms.vstring('photon_eb_ECALonly_lowpt_var', 'photon_eb_ECALonly_var', 'photon_ee_ECALonly_lowpt_var', 'photon_ee_ECALonly_var'),
                                          ),

              ecalrechitsEB       = cms.InputTag("reducedEcalRecHitsEB"),
              ecalrechitsEE       = cms.InputTag("reducedEcalRecHitsEE"),

              lowEnergy_ECALonlyThr = cms.double(300.),
              lowEnergy_ECALTRKThr = cms.double(50.),
              highEnergy_ECALTRKThr = cms.double(200.),
              eOverP_ECALTRKThr = cms.double(0.025),
              epDiffSig_ECALTRKThr = cms.double(15.),
              epSig_ECALTRKThr = cms.double(10.),

              )
    
