import FWCore.ParameterSet.Config as cms

# clustering sequence
from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.preshowerClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.dynamicHybridClusteringSequence_cff import *

hiEcalClusteringSequence = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*preshowerClusteringSequence*dynamicHybridClusteringSequence*multi5x5ClusteringSequence*multi5x5PreshowerClusteringSequence)

# reco photon producer
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photons.primaryVertexProducer = cms.string('pixel3Vertices') # replace the primary vertex
photonCore.scHybridBarrelProducer = cms.InputTag("correctedIslandBarrelSuperClusters") # use island for the moment
photonCore.scIslandEndcapProducer = cms.InputTag("correctedIslandEndcapSuperClusters") # use island for the moment
hiPhotonSequence = cms.Sequence(photonSequence)

# HI Egamma Isolation
from RecoHI.HiEgammaAlgos.HiEgammaIsolation_cff import *

# HI Ecal reconstruction
hiEcalClusters = cms.Sequence(hiEcalClusteringSequence)
hiEgammaSequence = cms.Sequence(hiPhotonSequence)
hiEgammaIsolationSequence = cms.Sequence(hiEgammaSequence * hiEgammaIsolationSequence)

# Test
