import FWCore.ParameterSet.Config as cms
import re
import itertools

from FWCore.ParameterSet.MassReplace import massReplaceInputTag
from HeterogeneousCore.AlpakaCore.functions import *
from HLTrigger.Configuration.common import *

def customizeHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel and Strip Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTDoLocalPixelSequence'):
        return process

    process.hltSiStripRawToClustersFacility = cms.EDProducer("SiStripClusterizerFromRaw",
    Algorithms = cms.PSet(
        CommonModeNoiseSubtractionMode = cms.string('Median'),
        PedestalSubtractionFedMode = cms.bool(True),
        SiStripFedZeroSuppressionMode = cms.uint32(4),
        TruncateInSuppressor = cms.bool(True),
        Use10bitsTruncation = cms.bool(False),
        doAPVRestore = cms.bool(False),
        useCMMeanMap = cms.bool(False)
    ),
    Clusterizer = cms.PSet(
        Algorithm = cms.string('ThreeThresholdAlgorithm'),
        ChannelThreshold = cms.double(2.0),
        ClusterThreshold = cms.double(5.0),
        ConditionsLabel = cms.string(''),
        MaxAdjacentBad = cms.uint32(0),
        MaxClusterSize = cms.uint32(8),
        MaxSequentialBad = cms.uint32(1),
        MaxSequentialHoles = cms.uint32(0),
        RemoveApvShots = cms.bool(True),
        SeedThreshold = cms.double(3.0),
        clusterChargeCut = cms.PSet(
            refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone')
        ),
        setDetId = cms.bool(True)
    ),
    ConditionsLabel = cms.string(''),
    DoAPVEmulatorCheck = cms.bool(False),
    HybridZeroSuppressed = cms.bool(False),
    LegacyUnpacker = cms.bool(False),
    ProductLabel = cms.InputTag("rawDataCollector"),
    onDemand = cms.bool(False)
    )
    
    
    process.hltSiStripMatchedRecHitsFull = cms.EDProducer( "SiStripRecHitConverter",
    ClusterProducer = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    rphiRecHits = cms.string( "rphiRecHit" ),
    stereoRecHits = cms.string( "stereoRecHit" ),
    matchedRecHits = cms.string( "matchedRecHit" ),
    useSiStripQuality = cms.bool( False ),
    MaskBadAPVFibers = cms.bool( False ),
    doMatching = cms.bool( True ),
    StripCPE = cms.ESInputTag( "hltESPStripCPEfromTrackAngle","hltESPStripCPEfromTrackAngle" ),
    Matcher = cms.ESInputTag( "SiStripRecHitMatcherESProducer","StandardMatcher" ),
    siStripQualityLabel = cms.ESInputTag( "","" )
    )
    
    process.hltSiPixelOnlyRecHitsSoA = cms.EDProducer('SiPixelRecHitAlpakaPhase1@alpaka',
        beamSpot = cms.InputTag('hltOnlineBeamSpotDevice'),
        src = cms.InputTag('hltSiPixelClustersSoA'),
        CPE = cms.string('PixelCPEFastParams'),
        mightGet = cms.optional.untracked.vstring,
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )
    
    process.hltSiPixelRecHitsSoA = cms.EDProducer('SiStripRecHitSoAPhase1@alpaka',
      stripRecHitSource = cms.InputTag('hltSiStripMatchedRecHitsFull', 'matchedRecHit'),
      beamSpot =  cms.InputTag('hltOnlineBeamSpot'),
      pixelRecHitSoASource = cms.InputTag('hltSiPixelOnlyRecHitsSoA'),
      mightGet = cms.optional.untracked.vstring,
      
      alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
      )
    )
    process.hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelOnlyRecHitsSoA'),
        src = cms.InputTag('hltSiPixelClusters'),
    )

    ###
    ### Sequence: Addition of strip modules in Pixel Local Reconstruction
    ###
    
    process.HLTDoLocalPixelSequence.insert(process.HLTDoLocalPixelSequence.index(process.hltSiPixelDigiErrors)+1,process.hltSiStripRawToClustersFacility+process.hltSiStripMatchedRecHitsFull+process.hltSiPixelOnlyRecHitsSoA)
    
    ###
    ### SerialSync version of Pixel Local Reconstruction
    ###

    process.hltSiPixelOnlyRecHitsSoACPUSerial = makeSerialClone(process.hltSiPixelOnlyRecHitsSoA,
        beamSpot = 'hltOnlineBeamSpotDeviceSerialSync',
        src = 'hltSiPixelClustersSoASerialSync'
    )
    process.hltSiPixelRecHitsSoASerialSync = makeSerialClone(process.hltSiPixelRecHitsSoA,
      pixelRecHitSoASource = cms.InputTag('hltSiPixelOnlyRecHitsSoACPUSerial'),
      )
    
    process.hltSiPixelRecHitsSerialSync = process.hltSiPixelRecHits.clone(
         pixelRecHitSrc = 'hltSiPixelOnlyRecHitsSoACPUSerial',
         src = 'hltSiPixelClustersSerialSync',
    )

    if(not hasattr(process,'hltSiPixelOnlyRecHitsSoA')):
        return process
    if not hasattr(process, 'HLTDoLocalPixelSequenceSerialSync'):
        return process
    process.HLTDoLocalPixelSequenceSerialSync.insert(process.HLTDoLocalPixelSequenceSerialSync.index(process.hltSiPixelDigiErrorsSerialSync)+1,process.hltSiStripRawToClustersFacility+process.hltSiStripMatchedRecHitsFull+process.hltSiPixelOnlyRecHitsSoACPUSerial)
    return process


def customizeHLTforAlpakaPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTRecoPixelTracksSequence'):
        return process
                
    for producer in producers_by_type(process, "TrackListMerger"):
        current_producers = producer.TrackProducers
        if (
            'hltIter0PFlowTrackSelectionHighPurity' in current_producers and 
            'hltDoubletRecoveryPFlowTrackSelectionHighPurity' in current_producers
        ):
            setattr(producer, "TrackProducers",cms.VInputTag('hltIter0PFlowTrackSelectionHighPurity'))
            setattr(producer,"hasSelector",cms.vint32( 0))
            setattr(producer,"indivShareFrac",cms.vdouble( 1.0))
            setattr(producer, "selectedTrackQuals", cms.VInputTag( 'hltIter0PFlowTrackSelectionHighPurity'))
            setattr(producer,"setsToMerge",cms.VPSet( cms.PSet(  pQual = cms.bool( False ), tLists = cms.vint32( 0))))
                    
    if hasattr(process, "hltDoubletRecoveryPFlowTrackSelectionHighPurity"):
        del process.hltDoubletRecoveryPFlowTrackSelectionHighPurity

    for producer in producers_by_type(process, "TrackListMerger"):
        current_producers = producer.TrackProducers
        if (
            'hltIter0PFlowTrackSelectionHighPuritySerialSync' in current_producers and
            'hltDoubletRecoveryPFlowTrackSelectionHighPuritySerialSync' in current_producers
        ):
            setattr(producer, "TrackProducers",cms.VInputTag('hltIter0PFlowTrackSelectionHighPuritySerialSync'))
            setattr(producer,"hasSelector",cms.vint32( 0))
            setattr(producer,"indivShareFrac",cms.vdouble( 1.0))
            setattr(producer, "selectedTrackQuals", cms.VInputTag( 'hltIter0PFlowTrackSelectionHighPuritySerialSync'))
            setattr(producer,"setsToMerge",cms.VPSet( cms.PSet(  pQual = cms.bool( False ), tLists = cms.vint32( 0))))
    if hasattr(process, "HLTIterativeTrackingDoubletRecovery"):
        del process.HLTIterativeTrackingDoubletRecovery
    if hasattr(process, "HLTIterativeTrackingDoubletRecoverySerialSync"):
        del process.HLTIterativeTrackingDoubletRecoverySerialSync
    if hasattr(process, "hltDoubletRecoveryPFlowTrackSelectionHighPuritySerialSync"):
        del process.hltDoubletRecoveryPFlowTrackSelectionHighPuritySerialSync
                    
    # alpaka EDProducer
    # consumes
    #  - TrackingRecHitsSoACollection<TrackerTraits>
    # produces
    #  - TkSoADevice
    
    process.frameSoAESProducerPhase1Strip = cms.ESProducer('FrameSoAESProducerPhase1Strip@alpaka',
      ComponentName = cms.string('FrameSoAPhase1Strip'),
      appendToDataLabel = cms.string(''),
      alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
      )
    )
    
    process.hltPixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase1Strip@alpaka',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoA'),
        frameSoA = cms.string('FrameSoAPhase1Strip'),
        ptmin = cms.double(0.9),
        maxNumberOfDoublets = cms.uint32(8*256*1024),
        CAThetaCutBarrel = cms.double(0.002),
        CAThetaCutForward = cms.double(0.003),
        hardCurvCut = cms.double(0.0328407225),
        dcaCutInnerTriplet = cms.double(0.15),
        dcaCutOuterTriplet = cms.double(0.25),
        CAThetaCutBarrelPixelBarrelStrip = cms.double(0.002),
        CAThetaCutBarrelPixelForwardStrip = cms.double(0.002),
        CAThetaCutBarrelStripForwardStrip = cms.double(0.002),
        CAThetaCutBarrelStrip = cms.double(0.002),
        CAThetaCutDefault = cms.double(0.003),
        dcaCutInnerTripletPixelStrip = cms.double(0.15),
        dcaCutOuterTripletPixelStrip = cms.double(0.25),
        dcaCutTripletStrip = cms.double(0.25),
        dcaCutTripletDefault = cms.double(0.25),
        earlyFishbone = cms.bool(True),
        lateFishbone = cms.bool(False),
        fillStatistics = cms.bool(False),
        minHitsPerNtuplet = cms.uint32(3),
        minHitsForSharingCut = cms.uint32(10),
        fitNas4 = cms.bool(False),
        doClusterCut = cms.bool(True),
        doZ0Cut = cms.bool(True),
	cellZ0Cut = cms.double(10.0),
        cellPtCut = cms.double(0.5),
        doPtCut = cms.bool(True),
        useRemovers = cms.bool(True),
        useRiemannFit = cms.bool(False),
        doSharedHitCut = cms.bool(True),                                                                                               
        dupPassThrough = cms.bool(False),                                                                                             
        useSimpleTripletCleaner = cms.bool(True),                                                                                     
        idealConditions = cms.bool(False),
        includeJumpingForwardDoublets = cms.bool(True),
        trackQualityCuts = cms.PSet(
          chi2MaxPt = cms.double(10),
          chi2Coeff = cms.vdouble(
            0.9,
            1.8
          ),
	chi2Scale = cms.double(8),
          tripletMinPt = cms.double(0.5),
          tripletMaxTip = cms.double(0.3),
          tripletMaxZip = cms.double(12),
          quadrupletMinPt = cms.double(0.3),
          quadrupletMaxTip = cms.double(0.5),
          quadrupletMaxZip = cms.double(12)
        ),
    phiCuts = cms.vint32(
     522, 730, 730, 522, 730, 626, 626, 522, 522, 522, 626, 522, 1200, 1200, 626, 730, 626, 626, 522, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,  2000,  2000, 626,  522, 2000, 2000, 2000, 2000),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltL2TauTagNNProducer'):
        process.hltL2TauTagNNProducer = cms.EDProducer("L2TauNNProducerAlpakaStrip", **process.hltL2TauTagNNProducer.parameters_())
    process.hltPixelTracksSoASerialSync = makeSerialClone(process.hltPixelTracksSoA,
        pixelRecHitSrc = 'hltSiPixelRecHitsSoASerialSync'
    )
    
    process.hltPixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase1Strip",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('loose'),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
        trackSrc = cms.InputTag("hltPixelTracksSoA"),
        useStripHits = cms.bool(True), 
        hitModuleStartSrc = cms.InputTag("hltSiPixelRecHitsSoA"),     
        stripRecHitLegacySrc = cms.InputTag('hltSiStripMatchedRecHitsFull', 'matchedRecHit'),
        mightGet = cms.optional.untracked.vstring
    )
    process.hltESPTTRHBuilderPixelOnly = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
        ComponentName = cms.string('hltESPTTRHBuilderPixelOnly'),
        ComputeCoarseLocalPositionFromDisk = cms.bool(False),
        Matcher = cms.string('StandardMatcher'),
        Phase2StripCPE = cms.string(''),
        PixelCPE = cms.string('hltESPPixelCPEGeneric'),
        StripCPE = cms.string('hltESPStripCPEfromTrackAngle'),
        appendToDataLabel = cms.string('')
    )
    process.hltPixelTracksSerialSync = process.hltPixelTracks.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsSerialSync"),
        hitModuleStartSrc = cms.InputTag("hltSiPixelRecHitsSoASerialSync"),
        trackSrc = cms.InputTag("hltPixelTracksSoASerialSync")
    )
    process.HLTRecoPixelTracksTask = cms.ConditionalTask(
        process.hltPixelTracksSoA,
        process.hltPixelTracks,
    )
    process.HLTRecoPixelTracksCPUSerialTask = cms.ConditionalTask(
        process.hltPixelTracksSoASerialSync,
        process.hltPixelTracksSerialSync,
    )
    process.HLTRecoPixelTracksSequence = cms.Sequence(process.hltPixelTracksSoA+process.hltPixelTracks)
    if hasattr(process, 'hltPixelTracksSerialSync'):
        process.HLTRecoPixelTracksCPUSerialSequence = cms.Sequence(process.hltPixelTracksSoASerialSync+process.hltPixelTracksSerialSync)

    return process

def customizeHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''
    
    # alpaka EDProducer
    # consumes
    #  - TkSoADevice
    # produces
    #  - ZVertexDevice
    
    process.hltPixelVerticesSoA = cms.EDProducer('PixelVertexProducerAlpakaPhase1Strip@alpaka',
        oneKernel = cms.bool(True),
        useDensity = cms.bool(True),
        useDBSCAN = cms.bool(False),
        useIterative = cms.bool(False),
        minT = cms.int32(2),
        eps = cms.double(0.07),
        errmax = cms.double(0.01),
        chi2max = cms.double(9),
        PtMin = cms.double(0.5),
        PtMax = cms.double(75),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoA'),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltPixelVerticesSoASerialSync = makeSerialClone(process.hltPixelVerticesSoA,
        pixelTrackSrc = 'hltPixelTracksSoASerialSync'
    )

    process.hltPixelVertices = cms.EDProducer("PixelVertexProducerFromSoAAlpaka",
        TrackCollection = cms.InputTag("hltPixelTracks"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltPixelVerticesSoA")
    )

    process.hltPixelVerticesSerialSync = process.hltPixelVertices.clone(
        TrackCollection = cms.InputTag("hltPixelTracksSerialSync"),
        src = cms.InputTag("hltPixelVerticesSoASerialSync")
    )

    if hasattr(process, 'hltPixelVerticesCPU'):
        del process.hltPixelVerticesCPU
    if hasattr(process, 'hltPixelVerticesCPUOnly'):
        del process.hltPixelVerticesCPUOnly
    if hasattr(process, 'hltPixelVerticesFromGPU'):
        del process.hltPixelVerticesFromGPU
    if hasattr(process, 'hltPixelVerticesGPU'):
        del process.hltPixelVerticesGPU

    ## failsafe for fake menus
    if hasattr(process, 'hltTrimmedPixelVertices'):
        process.HLTRecopixelvertexingTask = cms.ConditionalTask(
            process.HLTRecoPixelTracksTask,
            process.hltPixelVerticesSoA,
            process.hltPixelVertices,
            process.hltTrimmedPixelVertices
        )
        process.HLTRecopixelvertexingSequence = cms.Sequence( process.HLTRecopixelvertexingTask )
    if hasattr(process, 'hltTrimmedPixelVerticesSerialSync'):
        process.HLTRecopixelvertexingCPUSerialTask = cms.ConditionalTask(
            process.HLTRecoPixelTracksCPUSerialTask,
            process.hltPixelVerticesSoASerialSync,
            process.hltPixelVerticesSerialSync,
            process.hltTrimmedPixelVerticesSerialSync
        )
        process.HLTRecopixelvertexingSequenceSerialSync = cms.Sequence( process.HLTRecopixelvertexingCPUSerialTask )

    return process

def customizeHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''
    process = customizeHLTforAlpakaPixelRecoLocal(process)
    process = customizeHLTforAlpakaPixelRecoTracking(process)
    process = customizeHLTforAlpakaPixelRecoVertexing(process)

    return process


def customizeHLTforAlpakaStripNoDoubletRecovery(process):
    process = customizeHLTforAlpakaPixelReco(process)

    return process
