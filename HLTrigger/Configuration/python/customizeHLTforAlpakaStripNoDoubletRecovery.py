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
                    
    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2 = cms.vdouble( 999.0, 25.0, 99.0 )
    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2n = cms.vdouble( 1.2, 1.0, 999.0 )
    process.hltIter0PFlowTrackCutClassifier.mva.dr_par = cms.PSet(
        d0err = cms.vdouble(0.003, 0.003, 0.003),
        d0err_par = cms.vdouble(0.001, 0.001, 0.001),
        dr_exp = cms.vint32(4, 4, 4),
        dr_par1 = cms.vdouble(3.40282346639e+38, 0.8, 3.40282346639e+38),
        dr_par2 = cms.vdouble(3.40282346639e+38, 0.6, 3.40282346639e+38)
    ),
    process.hltIter0PFlowTrackCutClassifier.mva.dz_par = cms.PSet(
        dz_exp = cms.vint32(4, 4, 4),
        dz_par1 = cms.vdouble(3.40282346639e+38, 0.75, 3.40282346639e+38),
        dz_par2 = cms.vdouble(3.40282346639e+38, 0.5, 3.40282346639e+38)
    )
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

    def remove_doublet_recovery(process, sequence_name):
        # Remove from the sequence
        seq = getattr(process, sequence_name)
        processes_to_remove = [p for p in seq.moduleNames() if p.startswith('hltDoubletRecovery')]
        for process_name in processes_to_remove:
            seq.remove(getattr(process, process_name))
            
        # Remove from the execution path
        for path_name in process.paths:
            path = getattr(process, path_name)
            for process_name in processes_to_remove:
                if process_name in path.moduleNames():
                    path.remove(getattr(process, process_name))
    
        # Also remove from the process itself to ensure they're not executed
        for process_name in processes_to_remove:
            if hasattr(process, process_name):
                delattr(process, process_name)

        return process

    process = remove_doublet_recovery(process, 'HLTIterativeTrackingIter02')
    process.hltIter0PFlowCkfTrackCandidates.maxNSeeds = cms.uint32(32*4*1024)


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
        useRemovers = cms.bool(False),
        useRiemannFit = cms.bool(False),
        doSharedHitCut = cms.bool(False),                                                                                               
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

    process.hltPixelTracksSoA.useRemovers =     False
    process.hltPixelTracksSoA.doSharedHitCut = False
    process.hltPixelTracksSoA.CAThetaCutBarrel = cms.double(0.00111685053)
    process.hltPixelTracksSoA.CAThetaCutForward = cms.double(0.00249872683)
    process.hltPixelTracksSoA.hardCurvCut =  cms.double(0.695091509)
    process.hltPixelTracksSoA.dcaCutInnerTriplet = cms.double(0.0419242041)
    process.hltPixelTracksSoA.dcaCutOuterTriplet = cms.double(0.293522194)

    return process

def customizeHLTforAlpakaPixelRecoVertexing(process):
    # Iterate over all producers of the specific type
    for prod in producers_by_type(process, "PixelVertexProducerAlpakaPhase1@alpaka"):
        # Change the type of the producer to the new type
        prod._TypedParameterizable__type = "PixelVertexProducerAlpakaPhase1Strip@alpaka"
    
    for prod in producers_by_type(process, "alpaka_serial_sync::PixelVertexProducerAlpakaPhase1"):
        # Change the type of the producer to the new type
        prod._TypedParameterizable__type = "alpaka_serial_sync::PixelVertexProducerAlpakaPhase1Strip"

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
