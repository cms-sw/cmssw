import FWCore.ParameterSet.Config as cms

def customiseForRunI(process):

    # apply only in reco step
    if not hasattr(process,'reconstruction'):
        return process

    # Put back 2012 default tracking. This piece of code is ugly.

    # first remove the current/default version of trackingGlocalReco
    # and delete all its descendent sequences that are going to be
    # redefined later on by the new process.load()

    # apply only in reco step
    if not hasattr(process,'reconstruction'):
        return process

    tgrIndex = process.globalreco.index(process.trackingGlobalReco)
    tgrIndexFromReco = process.reconstruction_fromRECO.index(process.InitialStep)
    process.globalreco.remove(process.trackingGlobalReco)
    process.reconstruction_fromRECO.remove(process.InitialStep)
    process.reconstruction_fromRECO.remove(process.DetachedTripletStep)
    process.reconstruction_fromRECO.remove(process.LowPtTripletStep)
    process.reconstruction_fromRECO.remove(process.PixelPairStep)
    process.reconstruction_fromRECO.remove(process.MixedTripletStep)
    process.reconstruction_fromRECO.remove(process.PixelLessStep)
    process.reconstruction_fromRECO.remove(process.TobTecStep)
    process.reconstruction_fromRECO.remove(process.JetCoreRegionalStep)
    process.reconstruction_fromRECO.remove(process.earlyGeneralTracks)
    process.reconstruction_fromRECO.remove(process.muonSeededStep)
    process.reconstruction_fromRECO.remove(process.preDuplicateMergingGeneralTracks)
    process.reconstruction_fromRECO.remove(process.generalTracksSequence)
    process.reconstruction_fromRECO.remove(process.ConvStep)
    process.reconstruction_fromRECO.remove(process.conversionStepTracks)
    del process.trackingGlobalReco
    del process.ckftracks
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.ckftracks_woBH
    del process.iterTracking
    del process.InitialStep
    del process.LowPtTripletStep
    del process.PixelPairStep
    del process.DetachedTripletStep
    del process.MixedTripletStep
    del process.PixelLessStep
    del process.TobTecStep
    del process.JetCoreRegionalStep

    # Load the new Iterative Tracking configuration
    process.load("RecoTracker.Configuration.RecoTrackerRunI_cff")

    process.globalreco.insert(tgrIndex, process.trackingGlobalReco)
    process.globalreco.insert(tgrIndex, process.recopixelvertexing)
    process.reconstruction_fromRECO.insert(tgrIndexFromReco, process.iterTracking)

    # Now get rid of spurious reference to JetCore step
    process.earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                     'lowPtTripletStepTracks',
                                     'pixelPairStepTracks',
                                     'detachedTripletStepTracks',
                                     'mixedTripletStepTracks',
                                     'pixelLessStepTracks',
                                     'tobTecStepTracks'
                                     ]

    process.earlyGeneralTracks.inputClassifiers =["initialStepSelector",
                                      "lowPtTripletStepSelector",
                                      "pixelPairStepSelector",
                                      "detachedTripletStep",
                                      "mixedTripletStep",
                                      "pixelLessStepSelector",
                                      "tobTecStep"
                                      ]

    # Now get rid of any pre-splitting business
    process.siPixelClusters = process.siPixelClustersPreSplitting.clone()
    process.pixeltrackerlocalreco.replace(process.siPixelClustersPreSplitting, process.siPixelClusters)
    process.pixeltrackerlocalreco.replace(process.siPixelRecHitsPreSplitting, process.siPixelRecHits)
    process.clusterSummaryProducer.pixelClusters = 'siPixelClusters'
    process.globalreco.replace(process.MeasurementTrackerEventPreSplitting, process.MeasurementTrackerEvent)
    process.globalreco.replace(process.siPixelClusterShapeCachePreSplitting, process.siPixelClusterShapeCache)

    # Now restore pixelVertices wherever was not possible with an ad-hoc RunI cfg
    process.muonSeededTracksInOutClassifier.vertices = 'pixelVertices'
    process.muonSeededTracksOutInClassifier.vertices = 'pixelVertices'
    process.muonSeededTracksOutInDisplacedClassifier.vertices = 'pixelVertices'
    process.duplicateTrackClassifier.vertices = 'pixelVertices'
    process.duplicateDisplacedTrackClassifier.vertices = 'pixelVertices'
    process.convStepSelector.vertices = 'pixelVertices'
    process.pixelPairElectronSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = 'pixelVertices'
    process.ak4CaloJetsForTrk.srcPVs = 'pixelVertices'
    process.photonConvTrajSeedFromSingleLeg.primaryVerticesTag = 'pixelVertices'

    # ... and finally turn off all possible references to CCC: this is
    # done by switching off the Tight and Loose reftoPSet, rather than
    # following all the places in which they are effectively used in
    # release. The RunI-like tracking already uses CCCNone: this will
    # be useful mainly for conversions.
    process.SiStripClusterChargeCutTight.value = -1.
    process.SiStripClusterChargeCutLoose.value = -1.


    if hasattr(process, 'TrackingDQMSourceTier0Common') and 'TrackSeedMonjetCoreRegionalStep' in process.TrackingDQMSourceTier0Common.moduleNames():
          process.TrackingDQMSourceTier0Common.remove(process.TrackSeedMonjetCoreRegionalStep)

    if hasattr(process, 'TrackingDQMSourceTier0') and 'TrackSeedMonjetCoreRegionalStep' in process.TrackingDQMSourceTier0.moduleNames():
           process.TrackingDQMSourceTier0.remove(process.TrackSeedMonjetCoreRegionalStep)

    if hasattr(process, 'TrackingDQMSourceTier0MinBias') and 'TrackSeedMonjetCoreRegionalStep' in process.TrackingDQMSourceTier0MinBias.moduleNames():
          process.TrackingDQMSourceTier0MinBias.remove(process.TrackSeedMonjetCoreRegionalStep)



    return process
