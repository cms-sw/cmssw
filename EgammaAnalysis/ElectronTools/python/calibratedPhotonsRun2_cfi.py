
import FWCore.ParameterSet.Config as cms

correctionType = "Prompt2015"
files = {"Prompt2015":"EgammaAnalysis/ElectronTools/data/74X_Prompt_2015",
         "76XReReco" :"EgammaAnalysis/ElectronTools/data/76X_16DecRereco_2015"}

calibratedPatPhotons = cms.EDProducer("CalibratedPatPhotonProducerRun2",

                                      # input collections
                                      photons = cms.InputTag('slimmedPhotons'),
                                      
                                      # data or MC corrections
                                      # if isMC is false, data corrections are applied
                                      isMC = cms.bool(False),
                                      
                                      # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                      isSynchronization = cms.bool(False),

                                      correctionFile = cms.string(files[correctionType])
                                      )

calibratedPhotons = cms.EDProducer("CalibratedPhotonProducerRun2",

                                   # input collections
                                   photons = cms.InputTag('photons'),
                                   
                                   # data or MC corrections
                                   # if isMC is false, data corrections are applied
                                   isMC = cms.bool(False),
                                   
                                   # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                   isSynchronization = cms.bool(False),

                                   correctionFile = cms.string(files[correctionType])
                                   )


