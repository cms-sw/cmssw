import FWCore.ParameterSet.Config as cms

#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
allConversionTask = cms.Task(allConversions)
allConversionSequence = cms.Sequence(allConversionTask)

allConversionsOldEG = allConversions.clone(
    scBarrelProducer   = "correctedHybridSuperClusters",
    bcBarrelCollection = "hybridSuperClusters:hybridBarrelBasicClusters",
    scEndcapProducer   = "correctedMulti5x5SuperClustersWithPreshower",
    bcEndcapCollection = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
)
allConversionOldEGSequence = cms.Sequence(allConversionsOldEG)

