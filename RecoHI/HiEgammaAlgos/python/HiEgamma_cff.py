import FWCore.ParameterSet.Config as cms

# clustering sequence
from RecoHI.HiEgammaAlgos.HiIslandClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.preshowerClusteringSequence_cff import *
from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import *

particleFlowSuperClusterECAL.regressionConfig.vertexCollection = 'hiSelectedVertex'

hiEcalClusteringSequence = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*multi5x5ClusteringSequence*multi5x5PreshowerClusteringSequence*preshowerClusteringSequence*particleFlowSuperClusteringSequence)


# reco photon producer
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *

# use island for the moment
photonCore.scHybridBarrelProducer = cms.InputTag("correctedIslandBarrelSuperClusters")
photonCore.scIslandEndcapProducer = cms.InputTag("correctedIslandEndcapSuperClusters")
photonCore.minSCEt = cms.double(8.0)
photons.minSCEtBarrel = cms.double(5.0)
photons.minSCEtEndcap = cms.double(15.0)
photons.minR9Barrel = cms.double(10.)  #0.94
photons.minR9Endcap = cms.double(10.)   #0.95
photons.maxHoverEEndcap = cms.double(0.5)  #0.5
photons.maxHoverEBarrel = cms.double(0.99)  #0.5
photons.primaryVertexProducer = cms.InputTag('hiSelectedVertex') # replace the primary vertex
photons.isolationSumsCalculatorSet.trackProducer = isolationInputParameters.track    # cms.InputTag("highPurityTracks")

from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducer
hiPhotonSequence = cms.Sequence(photonSequence * photonIsolationHIProducer)

# HI Ecal reconstruction
hiEcalClusters = cms.Sequence(hiEcalClusteringSequence)
hiEgammaSequence = cms.Sequence(hiPhotonSequence)

# HI Spike Clean Sequence
import RecoHI.HiEgammaAlgos.hiSpikeCleaner_cfi
hiSpikeCleanedSC = RecoHI.HiEgammaAlgos.hiSpikeCleaner_cfi.hiSpikeCleaner.clone()
cleanPhotonCore = photonCore.clone(
    scHybridBarrelProducer = cms.InputTag("hiSpikeCleanedSC")
)
cleanPhotons = photons.clone(
    photonCoreProducer = cms.InputTag("cleanPhotonCore")
)

hiPhotonCleaningSequence = cms.Sequence(hiSpikeCleanedSC *
                                        cleanPhotonCore  *
                                        cleanPhotons)
