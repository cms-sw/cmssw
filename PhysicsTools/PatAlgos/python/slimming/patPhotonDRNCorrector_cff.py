import FWCore.ParameterSet.Config as cms 
from RecoEgamma.EgammaTools.patPhotonDRNCorrectionProducer_cfi import patPhotonDRNCorrectionProducer

patPhotonsDRN = patPhotonDRNCorrectionProducer.clone(
                            particleSource = 'selectedPatPhotons',
                            rhoName = 'fixedGridRhoFastjetAll',
                            Client = dict(
                              modelName = 'photonObjectCombined',
                              modelConfigPath = 'RecoEgamma/EgammaPhotonProducers/data/models/photonObjectCombined/config.pbtxt',
                              timeout = 10
                            )
    )
