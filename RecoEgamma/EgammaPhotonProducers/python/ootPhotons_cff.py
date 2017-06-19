import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import gedPhotons as _gedPhotons

ootPhotons = _gedPhotons.clone(
    photonProducer = 'ootPhotonCore',
    candidateP4type = "fromEcalEnergy",
    reconstructionStep = "oot",
    pfEgammaCandidates = "",
    valueMapPhotons = ""
    )
del ootPhotons.regressionConfig
