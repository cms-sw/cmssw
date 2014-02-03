import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
from RecoEgamma.PhotonIdentification.photonId_cfi import *
# photonID sequence
photonIDSequence = cms.Sequence(PhotonIDProd)

