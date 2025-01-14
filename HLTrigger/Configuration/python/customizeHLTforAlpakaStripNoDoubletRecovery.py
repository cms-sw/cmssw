import FWCore.ParameterSet.Config as cms
import re
import itertools

from FWCore.ParameterSet.MassReplace import massReplaceInputTag
from HeterogeneousCore.AlpakaCore.functions import *
from HLTrigger.Configuration.common import *

def customizeHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTDoLocalPixelSequence'):
        return process

    process.hltESPSiPixelCablingSoA = cms.ESProducer('SiPixelCablingSoAESProducer@alpaka',
        CablingMapLabel = cms.string(''),
        UseQualityInfo = cms.bool(False),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPSiPixelGainCalibrationForHLTSoA = cms.ESProducer('SiPixelGainCalibrationForHLTSoAESProducer@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPPixelCPEFastParamsPhase1 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase1@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltESPPixelCPEFast'):
        del process.hltESPPixelCPEFast

    # alpaka EDProducer
    # consumes
    #  - reco::BeamSpot
    # produces
    #  - BeamSpotDevice
    process.hltOnlineBeamSpotDevice = cms.EDProducer('BeamSpotDeviceProducer@alpaka',
        src = cms.InputTag('hltOnlineBeamSpot'),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltOnlineBeamSpotToGPU'):
        # hltOnlineBeamSpotToGPU is currently still used in HIon menu,
        # remove it only if the relevant ConditionalTask of the HIon menu is not present
        # (this check mainly applies to the HLT combined table)
        if not (hasattr(process, 'HLTDoLocalPixelPPOnAATask') and process.HLTDoLocalPixelPPOnAATask.contains(process.hltOnlineBeamSpotToGPU)):
            del process.hltOnlineBeamSpotToGPU

    # alpaka EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces (* optional)
    #  - SiPixelClustersSoA
    #  - SiPixelDigisSoACollection
    #  - SiPixelDigiErrorsSoACollection *
    #  - SiPixelFormatterErrors *
    process.hltSiPixelClustersSoA = cms.EDProducer('SiPixelRawToClusterPhase1@alpaka',
        IncludeErrors = cms.bool(True),
        UseQualityInfo = cms.bool(False),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        VCaltoElectronGain      = cms.double(1),  # all gains=1, pedestals=0
        VCaltoElectronGain_L1   = cms.double(1),
        VCaltoElectronOffset    = cms.double(0),
        VCaltoElectronOffset_L1 = cms.double(0),
        InputLabel = cms.InputTag('rawDataCollector'),
        Regions = cms.PSet(),
        CablingMapLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltSiPixelClustersGPU'):
        del process.hltSiPixelClustersGPU

    process.hltSiPixelClusters = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase1',
        src = cms.InputTag('hltSiPixelClustersSoA'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
    )

    # used only in the PPRef menu for the legacy pixel track reconstruction
    process.hltSiPixelClustersCache = cms.EDProducer('SiPixelClusterShapeCacheProducer',
        src = cms.InputTag('hltSiPixelClusters'),
        onDemand = cms.bool(False)
    )

    # legacy EDProducer
    # consumes
    #  - SiPixelDigiErrorsHost
    #  - SiPixelFormatterErrors
    # produces
    #  - edm::DetSetVector<SiPixelRawDataError>
    #  - DetIdCollection
    #  - DetIdCollection, 'UserErrorModules'
    #  - edmNew::DetSetVector<PixelFEDChannel>
    process.hltSiPixelDigiErrors = cms.EDProducer('SiPixelDigiErrorsFromSoAAlpaka',
        digiErrorSoASrc = cms.InputTag('hltSiPixelClustersSoA'),
        fmtErrorsSoASrc = cms.InputTag('hltSiPixelClustersSoA'),
        CablingMapLabel = cms.string(''),
        UsePhase1 = cms.bool(True),
        ErrorList = cms.vint32(29),
        UserErrorList = cms.vint32(40)
    )
    # alpaka EDProducer
    # consumes
    #  - BeamSpotDevice
    #  - SiPixelClustersSoA
    #  - SiPixelDigisSoACollection
    # produces
    #  - TrackingRecHitsSoACollection<TrackerTraits>
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
        # autoselect the alpaka backend
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
    ### Task: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDevice,
        process.hltSiPixelClustersSoA,
        process.hltSiPixelClusters,   
        process.hltSiPixelClustersCache,          
        process.hltSiPixelDigiErrors, 
        process.hltSiStripRawToClustersFacility,
        process.hltSiStripMatchedRecHitsFull,
        process.hltSiPixelOnlyRecHitsSoA,
        process.hltSiPixelRecHitsSoA,
        process.hltSiPixelRecHits, 
    )
    process.HLTDoLocalPixelSequence = cms.Sequence(
        process.hltOnlineBeamSpotDevice+
        process.hltSiPixelClustersSoA+
        process.hltSiPixelClusters+                                                                                                                     
        process.hltSiPixelClustersCache+                                                                                                               
        process.hltSiPixelDigiErrors+                                                                                                                   
        process.hltSiStripRawToClustersFacility+
        process.hltSiStripMatchedRecHitsFull+
        process.hltSiPixelOnlyRecHitsSoA+
        process.hltSiPixelRecHitsSoA+
        process.hltSiPixelRecHits                                                                                                                                      
    )
            
    ###
    ### SerialSync version of Pixel Local Reconstruction
    ###
    process.hltOnlineBeamSpotDeviceSerialSync = makeSerialClone(process.hltOnlineBeamSpotDevice)

    process.hltSiPixelClustersSoASerialSync = makeSerialClone(process.hltSiPixelClustersSoA)

    process.hltSiPixelClustersSerialSync = process.hltSiPixelClusters.clone(
        src = 'hltSiPixelClustersSoASerialSync'
    )

    process.hltSiPixelDigiErrorsSerialSync = process.hltSiPixelDigiErrors.clone(
        digiErrorSoASrc = 'hltSiPixelClustersSoASerialSync',
        fmtErrorsSoASrc = 'hltSiPixelClustersSoASerialSync',
    )
    
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
    
    process.HLTDoLocalPixelCPUSerialTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDeviceSerialSync,
        process.hltSiPixelClustersSoASerialSync,
        process.hltSiPixelClustersSerialSync,
        process.hltSiPixelDigiErrorsSerialSync,
        process.hltSiStripRawToClustersFacility,
        process.hltSiStripMatchedRecHitsFull,
        process.hltSiPixelOnlyRecHitsSoACPUSerial,
        process.hltSiPixelRecHitsSoASerialSync,
        process.hltSiPixelRecHitsSerialSync
    )
    if not hasattr(process, 'HLTDoLocalPixelSequenceSerialSync'):
        return process
    process.HLTDoLocalPixelSequenceSerialSync = cms.Sequence( process.HLTDoLocalPixelCPUSerialTask)
    process.HLTDoLocalPixelSequenceSerialSync = cms.Sequence(
        process.hltOnlineBeamSpotDeviceSerialSync+
        process.hltSiPixelClustersSoASerialSync+
        process.hltSiPixelClustersSerialSync+
        process.hltSiPixelDigiErrorsSerialSync+
        process.hltSiStripRawToClustersFacility+
        process.hltSiStripMatchedRecHitsFull+
        process.hltSiPixelOnlyRecHitsSoACPUSerial+
        process.hltSiPixelRecHitsSoASerialSync+
        process.hltSiPixelRecHitsSerialSync
        )
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
        maxNumberOfDoublets = cms.uint32(4*256*1024),
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
        doSharedHitCut = cms.bool(True), #originall True                                                                                              
        dupPassThrough = cms.bool(False), #originall False                                                                                             
        useSimpleTripletCleaner = cms.bool(True),#originally True,                                                                                     
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
     522, 730, 730, 522, 730, 626, 626, 522, 522, 522, 626, 522, 1200, 1200, 626, 730, 626, 626, 522, 2000, 2000, 2000, 2000,  2000, 2000, 2000, 2000, 2000, 2000, 2000,  2000,  2000, 626,  522, 2000, 2000, 2000, 2000),

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
                
    process.HLTRecoPixelTracksTask = cms.ConditionalTask(
        process.hltPixelTracksSoA,
        process.hltPixelTracks,
    )

    process.hltPixelTracksSerialSync = process.hltPixelTracks.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsSerialSync"),
        hitModuleStartSrc = cms.InputTag("hltSiPixelRecHitsSoASerialSync"),     
        trackSrc = cms.InputTag("hltPixelTracksSoASerialSync")
    )

    process.HLTRecoPixelTracksCPUSerialTask = cms.ConditionalTask(
        process.hltPixelTracksSoASerialSync,
        process.hltPixelTracksSerialSync,
    )
    process.HLTRecoPixelTracksSequence = cms.Sequence( process.HLTRecoPixelTracksTask )
    process.HLTRecoPixelTracksCPUSerialSequence = cms.Sequence( process.HLTRecoPixelTracksCPUSerialTask )

    process.HLTRecoPixelTracksSequence = cms.Sequence(process.hltPixelTracksSoA+process.hltPixelTracks)
    if hasattr(process, 'hltPixelTracksSerialSync'):
        process.HLTRecoPixelTracksCPUSerialSequence = cms.Sequence(process.hltPixelTracksSoASerialSync+process.hltPixelTracksSerialSync)

    return process

def customizeHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''

    #if not hasattr(process, 'HLTRecopixelvertexingSequence'):
    #    return process
#
    ## do not apply the customisation if the menu is already using the alpaka pixel reconstruction
    #for prod in producers_by_type(process, 'PixelVertexProducerAlpakaPhase1Strip@alpaka'):
    #    return process

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


def customizeHLTforAlpakaStatus(process):

    if not hasattr(process, 'statusOnGPU'):
        return process

    process.hltBackend = cms.EDProducer('AlpakaBackendProducer@alpaka')

    insert_modules_before(process, process.statusOnGPU, process.hltBackend)

    del process.statusOnGPU

    process.hltStatusOnGPUFilter = cms.EDFilter('AlpakaBackendFilter',
        producer = cms.InputTag('hltBackend', 'backend'),
        backends = cms.vstring('CudaAsync', 'ROCmAsync')
    )

    insert_modules_before(process, process.statusOnGPUFilter, process.hltStatusOnGPUFilter)
    insert_modules_before(process, ~process.statusOnGPUFilter, ~process.hltStatusOnGPUFilter)

    del process.statusOnGPUFilter

    return process


def _replace_object(process, target, obj):
    for container in itertools.chain(
        process.sequences_().values(),
        process.paths_().values(),
        process.endpaths_().values()
    ):
        if target.label() in [bar for foo,bar in container.directDependencies()]:
            try:
                position = container.index(target)
                container.insert(position, obj)
                container.remove(target)
            except ValueError:
                container.associate(obj)
                container.remove(target)

    for container in itertools.chain(
        process.tasks_().values(),
        process.conditionaltasks_().values(),
    ):
        if target.label() in [bar for foo,bar in container.directDependencies()]:
            container.add(obj)
            container.remove(target)

    return process

def _rename_edmodule(process, oldModuleLabel, newModuleLabel, typeBlackList):
    if not hasattr(process, oldModuleLabel) or hasattr(process, newModuleLabel) or oldModuleLabel == newModuleLabel:
        return process
    oldObj = getattr(process, oldModuleLabel)
    if oldObj.type_() in typeBlackList:
        return process
    setattr(process, newModuleLabel, oldObj.clone())
    newObj = getattr(process, newModuleLabel)
    process = _replace_object(process, oldObj, newObj)
    process.__delattr__(oldModuleLabel)
    process = massReplaceInputTag(process, oldModuleLabel, newModuleLabel, False, True, False)
    for outputModuleLabel in process.outputModules_():
        outputModule = getattr(process, outputModuleLabel)
        if not hasattr(outputModule, 'outputCommands'):
            continue
        for outputCmdIdx, outputCmd in enumerate(outputModule.outputCommands):
            outputModule.outputCommands[outputCmdIdx] = outputCmd.replace(f'_{oldModuleLabel}_', f'_{newModuleLabel}_')
    return process

def _rename_edmodules(process, matchExpr, oldStr, newStr, typeBlackList):
    for moduleDict in [process.producers_(), process.filters_(), process.analyzers_()]:
        moduleLabels = list(moduleDict.keys())
        for moduleLabel in moduleLabels:
            if bool(re.match(matchExpr, moduleLabel)):
                moduleLabelNew = moduleLabel.replace(oldStr, '') + newStr
                process = _rename_edmodule(process, moduleLabel, moduleLabelNew, typeBlackList)
    return process

def _rename_container(process, oldContainerLabel, newContainerLabel):
    if not hasattr(process, oldContainerLabel) or hasattr(process, newContainerLabel) or oldContainerLabel == newContainerLabel:
        return process
    oldObj = getattr(process, oldContainerLabel)
    setattr(process, newContainerLabel, oldObj.copy())
    newObj = getattr(process, newContainerLabel)
    process = _replace_object(process, oldObj, newObj)
    process.__delattr__(oldContainerLabel)
    return process

def _rename_containers(process, matchExpr, oldStr, newStr):
    for containerName in itertools.chain(
        process.sequences_().keys(),
        process.tasks_().keys(),
        process.conditionaltasks_().keys()
    ):
        if bool(re.match(matchExpr, containerName)):
            containerNameNew = containerName.replace(oldStr, '') + newStr
            process = _rename_container(process, containerName, containerNameNew)
    return process

def customizeHLTforAlpakaRename(process):
    # mass renaming of EDModules and Sequences:
    # if the label matches matchRegex, remove oldStr and append newStr
    for matchRegex, oldStr, newStr in [
        [".*Portable.*", "Portable", ""],
        [".*SerialSync.*", "SerialSync", "SerialSync"],
        [".*CPUSerial.*", "CPUSerial", "SerialSync"],
        [".*CPUOnly.*", "CPUOnly", "SerialSync"],
    ]:
        process = _rename_edmodules(process, matchRegex, oldStr, newStr, ['HLTPrescaler'])
        process = _rename_containers(process, matchRegex, oldStr, newStr)

    return process


def customizeHLTforAlpakaStripNoDoubletRecoveryIncreseDoublets(process):
    print("applying AlpakaCustomizer")
    process = customizeHLTforAlpakaStatus(process)
    process = customizeHLTforAlpakaPixelReco(process)
    process = customizeHLTforAlpakaRename(process)

    return process
