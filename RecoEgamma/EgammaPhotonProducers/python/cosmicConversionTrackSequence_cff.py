import FWCore.ParameterSet.Config as cms

# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cfi import *
conversionTrackCandidates.scHybridBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
conversionTrackCandidates.scIslandEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
conversionTrackCandidates.bcBarrelCollection = cms.InputTag("cosmicBasicClusters","CosmicBarrelBasicClusters")
conversionTrackCandidates.bcEndcapCollection = cms.InputTag("cosmicBasicClusters","CosmicEndcapBasicClusters")

# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
ckfTracksFromConversionsTask = cms.Task(conversionTrackCandidates,ckfOutInTracksFromConversions,ckfInOutTracksFromConversions)
ckfTracksFromConversions = cms.Sequence(ckfTracksFromConversionsTask)
cosmicConversionTrackTask = cms.Task(ckfTracksFromConversionsTask)
cosmicConversionTrackSequence = cms.Sequence(cosmicConversionTrackTask)
