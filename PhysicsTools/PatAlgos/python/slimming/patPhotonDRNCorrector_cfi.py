import FWCore.ParameterSet.Config as cms 
from RecoEgamma.EgammaTools.patPhotonDRNCorrectionProducer_cfi import patPhotonDRNCorrectionProducer

patPhotonsDRN = patPhotonDRNCorrectionProducer.clone(
                            particleSource = 'selectedPatPhotons',
                            rhoName = 'fixedGridRhoFastjetAll',
                            Client = patPhotonDRNCorrectionProducer.Client.clone(
                              mode = 'Async',
                              Retry = cms.VPSet(
                                cms.PSet(
                                  retryType = cms.string('RetrySameServerAction'),
                                  allowedTries = cms.untracked.uint32(1)
                                )
                              ),
                              modelName = 'photonObjectCombined',
                              modelConfigPath = 'RecoEgamma/EgammaPhotonProducers/data/models/photonObjectCombined/config.pbtxt',
                              timeout = 10
                            )
    )
