import FWCore.ParameterSet.Config as cms

electronMVAValueMapProducer = cms.EDProducer('ElectronMVAValueMapProducer',
                                             # The module automatically detects AOD vs miniAOD, so we configure both
                                             #
                                             # AOD case
                                             #
                                             src = cms.InputTag('gedGsfElectrons'),
                                             #
                                             # miniAOD case
                                             #
                                             srcMiniAOD = cms.InputTag('slimmedElectrons'),
                                             #
                                             # MVA configurations
                                             #
                                             mvaPhys14NonTrigWeightFiles = cms.vstring
                                             ("RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_5_oldscenario2phys14_BDT.weights.xml",
                                              "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_5_oldscenario2phys14_BDT.weights.xml",
                                              "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_5_oldscenario2phys14_BDT.weights.xml",
                                              "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_10_oldscenario2phys14_BDT.weights.xml",
                                              "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_10_oldscenario2phys14_BDT.weights.xml",
                                              "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_10_oldscenario2phys14_BDT.weights.xml"
                                              )
                                             )
