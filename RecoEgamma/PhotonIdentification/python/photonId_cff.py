import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
from RecoEgamma.PhotonIdentification.photonId_cfi import *
# photonID sequence
PhotonIDProdGED = PhotonIDProd.clone(photonProducer = cms.string('gedPhotons'))
photonIDTask = cms.Task(PhotonIDProd)
photonIDSequence = cms.Sequence(photonIDTask)
photonIDTaskGED = cms.Task(PhotonIDProdGED)
photonIDSequenceGED = cms.Sequence(photonIDTaskGED)
