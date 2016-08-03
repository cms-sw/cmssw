import FWCore.ParameterSet.Config as cms

correctionType = "76XReReco"
files = {"Prompt2015":"EgammaAnalysis/ElectronTools/data/74X_Prompt_2015",
         "76XReReco" :"EgammaAnalysis/ElectronTools/data/76X_16DecRereco_2015"}

calibratedElectrons = cms.EDProducer("CalibratedElectronProducerRun2",

                                     # input collections
                                     electrons = cms.InputTag('gedGsfElectrons'),
                                     gbrForestName = cms.string("gedelectron_p4combination_25ns"),

                                     # data or MC corrections
                                     # if isMC is false, data corrections are applied
                                     isMC = cms.bool(False),
    
                                     # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                     isSynchronization = cms.bool(False),
                                     
                                     correctionFile = cms.string(files[correctionType])
                                     )


calibratedPatElectrons = cms.EDProducer("CalibratedPatElectronProducerRun2",
                                        
                                        # input collections
                                        electrons = cms.InputTag('slimmedElectrons'),
                                        gbrForestName = cms.string("gedelectron_p4combination_25ns"),
                                        
                                        # data or MC corrections
                                        # if isMC is false, data corrections are applied
                                        isMC = cms.bool(False),
                                        
                                        # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                        isSynchronization = cms.bool(False),

                                        correctionFile = cms.string(files[correctionType])
                                       )


