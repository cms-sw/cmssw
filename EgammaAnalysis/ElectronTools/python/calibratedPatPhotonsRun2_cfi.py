import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.ElectronTools.calibrationTablesRun2 import correctionType
from EgammaAnalysis.ElectronTools.calibrationTablesRun2 import files

calibratedPatPhotons = cms.EDProducer("CalibratedPatPhotonProducerRun2",

                                      # input collections
                                      photons = cms.InputTag('slimmedPhotons'),
                                      
                                      # data or MC corrections
                                      # if isMC is false, data corrections are applied
                                      isMC = cms.bool(False),
                                      autoDataType = cms.bool(True),
                                      
                                      # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
                                      isSynchronization = cms.bool(False),

                                      correctionFile = cms.string(files[correctionType]),
                                      recHitCollectionEB = cms.InputTag('reducedEgamma:reducedEBRecHits'),
                                      recHitCollectionEE = cms.InputTag('reducedEgamma:reducedEERecHits')

                                      )
