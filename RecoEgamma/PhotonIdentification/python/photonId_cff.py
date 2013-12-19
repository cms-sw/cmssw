import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
from RecoEgamma.PhotonIdentification.photonId_cfi import *
PhotonIDProdGED = PhotonIDProd.clone(photonProducer = cms.string('gedPhotons'))
# photonID sequence
photonIDSequenceGED = cms.Sequence(PhotonIDProdGED)
photonIDSequence = cms.Sequence(PhotonIDProd)
