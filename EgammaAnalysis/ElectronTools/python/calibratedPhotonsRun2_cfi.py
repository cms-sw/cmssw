
import FWCore.ParameterSet.Config as cms

correctionType = "80Xapproval"
files = {"Prompt2015":"EgammaAnalysis/ElectronTools/data/ScalesSmearings/74X_Prompt_2015",
         "76XReReco" :"EgammaAnalysis/ElectronTools/data/ScalesSmearings/76X_16DecRereco_2015_Etunc",
         "80Xapproval" : "EgammaAnalysis/ElectronTools/data/ScalesSmearings/80X_ichepV2_2016_pho"}

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


