import FWCore.ParameterSet.Config as cms

#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
allConversionSequence = cms.Sequence(allConversions)

allConversionsOldEG = allConversions.clone()
allConversionsOldEG.scBarrelProducer = cms.InputTag("correctedHybridSuperClusters")
allConversionsOldEG.bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
allConversionsOldEG.scEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
allConversionsOldEG.bcEndcapCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")

allConversionOldEGSequence = cms.Sequence(allConversionsOldEG)

