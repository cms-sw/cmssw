import FWCore.ParameterSet.Config as cms

regressionModifier = \
    cms.PSet( modifierName    = cms.string('EGExtraInfoModifierFromDB'),  

              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              useLocalFile     = cms.bool(False),
#              addressLocalFile = cms.FileInPath("regressionTrees.root"),
              
              electron_config = cms.PSet( # EB, EE
                                          regressionKey_ecalonly  = cms.vstring('electron_eb_ECALonly', 'electron_ee_ECALonly'),
                                          uncertaintyKey_ecalonly = cms.vstring('electron_eb_ECALonly_var', 'electron_ee_ECALonly_var'),
                                          regressionKey_ecaltrk  = cms.vstring('electron_eb_ECALTRK', 'electron_ee_ECALTRK'),
                                          uncertaintyKey_ecaltrk = cms.vstring('electron_eb_ECALTRK_var', 'electron_ee_ECALTRK_var'),
                                          ),
              
              photon_config   = cms.PSet( # EB, EE
                                          regressionKey_ecalonly  = cms.vstring('photon_eb_ECALonly', 'photon_ee_ECALonly'),
                                          uncertaintyKey_ecalonly = cms.vstring('photon_eb_ECALonly_var', 'photon_ee_ECALonly_var'),
                                          ),

              ecalrechitsEB       = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
              ecalrechitsEE       = cms.InputTag("ecalRecHit","EcalRecHitsEE")

              )
    
