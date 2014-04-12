import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
from RecoEgamma.PhotonIdentification.photonId_cfi import *
# photonID sequence
PhotonIDProdGED = PhotonIDProd.clone(photonProducer = cms.string('gedPhotons'))
photonIDSequence = cms.Sequence(PhotonIDProd)
photonIDSequenceGED = cms.Sequence(PhotonIDProdGED)
