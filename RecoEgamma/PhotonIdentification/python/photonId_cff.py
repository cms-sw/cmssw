import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
from RecoEgamma.PhotonIdentification.photonId_cfi import *
PhotonIDProdGED = PhotonIDProd.clone(photonProducer = cms.string('gedPhotons'))
# photonID sequence
photonIDSequence = cms.Sequence(PhotonIDProd)
photonIDSequenceGED = cms.Sequence(PhotonIDProdGED)
