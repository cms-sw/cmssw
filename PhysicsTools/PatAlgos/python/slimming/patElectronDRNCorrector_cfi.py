import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.patElectronDRNCorrectionProducer_cfi import patElectronDRNCorrectionProducer

patElectronsDRN = patElectronDRNCorrectionProducer.clone(
                            particleSource = 'selectedPatElectrons',
                            rhoName = 'fixedGridRhoFastjetAll',
                            Client = patElectronDRNCorrectionProducer.Client.clone(
                              mode = 'Async',
                              allowedTries = 1,
                              modelName = 'electronObjectEnsemble',
                              modelConfigPath = 'RecoEgamma/EgammaElectronProducers/data/models/electronObjectEnsemble/config.pbtxt',
                              timeout = 10
                            )
    )
