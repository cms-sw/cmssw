import FWCore.ParameterSet.Config as cms

# Conversion Track candidate producer 

from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cff import *
conversionTrackCandidates.scHybridBarrelProducer = "cosmicSuperClusters:CosmicBarrelSuperClusters"
conversionTrackCandidates.scIslandEndcapProducer = "cosmicSuperClusters:CosmicEndcapSuperClusters"
conversionTrackCandidates.bcBarrelCollection     = "cosmicBasicClusters:CosmicBarrelBasicClusters"
conversionTrackCandidates.bcEndcapCollection     = "cosmicBasicClusters:CosmicEndcapBasicClusters"

# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
ckfTracksFromConversionsTask = cms.Task(conversionTrackCandidates,ckfOutInTracksFromConversions,ckfInOutTracksFromConversions)
ckfTracksFromConversions = cms.Sequence(ckfTracksFromConversionsTask)
cosmicConversionTrackTask = cms.Task(ckfTracksFromConversionsTask)
cosmicConversionTrackSequence = cms.Sequence(cosmicConversionTrackTask)
