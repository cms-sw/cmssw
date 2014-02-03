import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.gedPhotonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import *

import RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi 

tmpGedPhotons = RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi.gedPhotons.clone()
tmpGedPhotons.photonProducer = cms.InputTag("gedPhotonCore")
tmpGedPhotons.outputPhotonCollection = cms.string("")
gedPhotonSequenceTmp = cms.Sequence(gedPhotonCore+tmpGedPhotons)


gedPhotons = RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi.gedPhotons.clone()
gedPhotons.photonProducer = cms.InputTag("tmpGedPhotons")
gedPhotons.outputPhotonCollection = cms.string("")
gedPhotonSequence    = cms.Sequence(gedPhotons)



