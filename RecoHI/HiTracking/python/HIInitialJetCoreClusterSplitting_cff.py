import FWCore.ParameterSet.Config as cms

### pixel primary vertices
from RecoHI.HiTracking.HIPixelVerticesPreSplitting_cff import *

#Jet Core emulation to identify jet-tracks
#modify the original hiAk4CaloJetsForTrkPreSplitting to hiAkPu4CaloJetsForTrkPreSplitting from HIJET reco
from RecoHI.HiJetAlgos.hiCaloJetsForTrk_cff import *
from RecoHI.HiTracking.hiJetCoreRegionalStep_cff import hiJetsForCoreTracking
hiCaloTowerForTrkPreSplitting = hiCaloTowerForTrk.clone()
hiAkPu4CaloJetsForTrkPreSplitting = akPu4CaloJetsForTrk.clone(
    src = 'hiCaloTowerForTrkPreSplitting',
    srcPVs = 'hiSelectedVertexPreSplitting'
)
hiAkPu4CaloJetsCorrectedPreSplitting = akPu4CaloJetsCorrected.clone(
    src = 'hiAkPu4CaloJetsForTrkPreSplitting'
)
hiAkPu4CaloJetsSelectedPreSplitting = akPu4CaloJetsSelected.clone(
    src = 'hiAkPu4CaloJetsCorrectedPreSplitting'
)
hiJetsForCoreTrackingPreSplitting = hiJetsForCoreTracking.clone(
    src = 'hiAkPu4CaloJetsSelectedPreSplitting'
)


from RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi import jetCoreClusterSplitter
siPixelClusters = jetCoreClusterSplitter.clone(
    pixelClusters = 'siPixelClustersPreSplitting',
    vertices      = 'hiSelectedVertexPreSplitting',
    cores         = 'hiJetsForCoreTrackingPreSplitting',
    deltaRmax     = 0.1,
    ptMin         = 50
)

from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import siPixelRecHits
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import MeasurementTrackerEvent
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
hiInitialJetCoreClusterSplittingTask = cms.Task(
                                hiPixelVerticesPreSplittingTask
                                , hiCaloTowerForTrkPreSplitting
                                , hiAkPu4CaloJetsForTrkPreSplitting
				, hiAkPu4CaloJetsCorrectedPreSplitting
				, hiAkPu4CaloJetsSelectedPreSplitting
                                , hiJetsForCoreTrackingPreSplitting
				, siPixelClusters
                                , siPixelRecHits
                                , MeasurementTrackerEvent
                                , siPixelClusterShapeCache)
