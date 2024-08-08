# adapted from  https://github.com/cms-sw/cmssw/blob/CMSSW_14_0_0/HLTrigger/Configuration/python/customizeHLTforAlpaka.py#L579
import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

def AlpakaHLTPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka.
    '''
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
    process.hltESPPixelCPEFastParamsPhase2 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2@alpaka', 
            ComponentName = cms.string("PixelCPEFastParamsPhase2"),
            appendToDataLabel = cms.string(''),
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
        )
    )
    process.hltPhase2OnlineBeamSpotDevice = cms.EDProducer('BeamSpotDeviceProducer@alpaka',
            src = cms.InputTag('hltOnlineBeamSpot'),
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
        )
    )
    process.hltPhase2SiPixelClustersSoA = cms.EDProducer('SiPixelPhase2DigiToCluster@alpaka',
        mightGet = cms.optional.untracked.vstring,
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )
    process.siPixelClusters = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase2',
        src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
    )
    process.siPixelClusterShapeCache = cms.EDProducer('SiPixelClusterShapeCacheProducer',
        src = cms.InputTag('siPixelClusters' ),
        onDemand = cms.bool( False )
    )
    process.hltPhase2SiPixelRecHitsSoA = cms.EDProducer('SiPixelRecHitAlpakaPhase2@alpaka',
        beamSpot = cms.InputTag('hltPhase2OnlineBeamSpotDevice'),
        src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
        CPE = cms.string('PixelCPEFastParamsPhase2'),
        mightGet = cms.optional.untracked.vstring,
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )
    process.siPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase2',
        pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
        src = cms.InputTag('siPixelClusters'),
    )

    process.HLTDoLocalPixelSequence = cms.Sequence(
         process.hltPhase2SiPixelClustersSoA
        +process.siPixelClusters
        +process.siPixelClusterShapeCache  # should we disable this? Still needed by tracker muons
        #+process.siPixelDigis             # not needed when copying digis from sim
        +process.hltPhase2SiPixelRecHitsSoA
        +process.siPixelRecHits
    )
    process.itLocalRecoSequence = cms.Sequence(
         process.HLTDoLocalStripSequence
        +process.HLTDoLocalPixelSequence
    )
    process.HLTBeamSpotSequence = cms.Sequence(
         process.hltOnlineBeamSpot
        +process.hltPhase2OnlineBeamSpotDevice
    )
    return process

def AlpakaHLTPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''
    # copied from https://github.com/cms-sw/cmssw/blob/CMSSW_14_1_X/Geometry/CommonTopologies/interface/SimplePixelTopology.h
    process.hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
        pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
        CPE = cms.string('PixelCPEFastParamsPhase2'),
        ptmin = cms.double(0.9),
        CAThetaCutBarrel = cms.double(0.002),
        CAThetaCutForward = cms.double(0.003),
        hardCurvCut = cms.double(0.0328407225),
        dcaCutInnerTriplet = cms.double(0.15),
        dcaCutOuterTriplet = cms.double(0.25),
        earlyFishbone = cms.bool(True),
        lateFishbone = cms.bool(False),
        fillStatistics = cms.bool(False),
        minHitsPerNtuplet = cms.uint32(4),
        phiCuts = cms.vint32(
            522, 522, 522, 626, 730, 730, 626, 730, 730, 522, 522,
            522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522,
            522, 522, 522, 522, 522, 522, 522, 730, 730, 730, 730,
            730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730,
            730, 730, 730, 522, 522, 522, 522, 522, 522, 522, 522
        ),
        maxNumberOfDoublets = cms.uint32(5*512*1024),
        minHitsForSharingCut = cms.uint32(10),
        fitNas4 = cms.bool(False),
        doClusterCut = cms.bool(True),
        doZ0Cut = cms.bool(True),
        doPtCut = cms.bool(True),
        useRiemannFit = cms.bool(False),
        doSharedHitCut = cms.bool(True),
        dupPassThrough = cms.bool(False),
        useSimpleTripletCleaner = cms.bool(True),
        idealConditions = cms.bool(False),
        includeJumpingForwardDoublets = cms.bool(True),
        trackQualityCuts = cms.PSet(
            # phase1 quality cuts not implemented for phase2
            # https://github.com/cms-sw/cmssw/blob/CMSSW_14_1_X/RecoTracker/PixelSeeding/plugins/CAHitNtupletGeneratorOnGPU.cc#L253-L257
            #chi2MaxPt = cms.double(10),
            #chi2Coeff = cms.vdouble(0.9, 1.8),
            #chi2Scale = cms.double(8),
            #tripletMinPt = cms.double(0.5),
            #tripletMaxTip = cms.double(0.3),
            #tripletMaxZip = cms.double(12),
            #quadrupletMinPt = cms.double(0.3),
            #quadrupletMaxTip = cms.double(0.5),
            #quadrupletMaxZip = cms.double(12)
            maxChi2 = cms.double(5.0),
            minPt   = cms.double(0.9),
            maxTip  = cms.double(0.3),
            maxZip  = cms.double(12.),
        ),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )
    process.hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase2",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('tight'),
        pixelRecHitLegacySrc = cms.InputTag("siPixelRecHits"),
        trackSrc = cms.InputTag("hltPhase2PixelTracksSoA")
    )
    process.hltPhase2PixelTracksSequence = cms.Sequence(
         process.HLTBeamSpotSequence
        +process.hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
        +process.hltPhase2PixelFitterByHelixProjections # needed by tracker muons
        +process.hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
        +process.hltPhase2PixelTracksSoA
        +process.hltPhase2PixelTracks
    )
    return process

def AlpakaHLTFullTracking(process):
    # original cfg at hand as in 14_1_0_pre4 simplified menu 
    # https://lguzzi.web.cern.ch/lguzzi/Tracking/Phase2/hlt-dump_simplifiedMenu_14_1_0_pre4.txt
    ''' track building seeded by hltPhase2PixelTracks. This configuration runs the initialStep
    and the highPtTripletStep
    '''
    process.HLTTrackingV61Sequence = cms.Sequence(
         process.itLocalRecoSequence
        +process.otLocalRecoSequence
        +process.trackerClusterCheck
        +process.hltPhase2PixelTracksSequence
        +process.hltPhase2PixelVertices
        +process.initialStepSequence
        +process.highPtTripletStepSequence
        +process.generalTracks
    )
    return process

def AlpakaHLTTrackingPath(process):
    ''' Create a tracking-only path
    '''
    process.HLT_AlpakaTrackingPath = cms.Path()
    process.HLT_AlpakaTrackingPath.insert(item=process.HLTBeginSequence         , index=len(process.HLT_AlpakaTrackingPath.moduleNames()))
    process.HLT_AlpakaTrackingPath.insert(item=process.HLTTrackingV61Sequence   , index=len(process.HLT_AlpakaTrackingPath.moduleNames()))
    process.HLT_AlpakaTrackingPath.insert(item=process.HLTEndSequence           , index=len(process.HLT_AlpakaTrackingPath.moduleNames()))
    process.outputmodule = cms.EndPath(process.FEVTDEBUGHLToutput)
    process.schedule.insert(-4, process.HLT_AlpakaTrackingPath)
    return process

def customizeHLTforAlpaka(process):
    ''' main customization function. Runs the full menu with tracking done
    in two iterations (initialStep and highPtTripletStep) using patatrack pixel tracks
    '''
    process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
    process.load('Configuration.StandardSequences.Accelerators_cff')
    process = AlpakaHLTPixelRecoLocal(process)
    process = AlpakaHLTPixelRecoTracking(process)
    process = AlpakaHLTFullTracking(process)
    return process

def customizeHLTforAlpakaSingleIteration(process):
    ''' main customization function. Runs the full menu with tracking done
    in a single iteration (initialStep) using patatrack pixel tracks
    '''
    process = customizeHLTforAlpaka(process)
    process.generalTracks = process.initialStepTrackSelectionHighPurity.clone()
    process.initialStepSequence.remove(process.initialStepTrackSelectionHighPurity)
    process.HLTTrackingV61Sequence.remove(process.hltPhase2PixelTracksAndHighPtStepTrackingRegions)
    process.HLTTrackingV61Sequence.remove(process.highPtTripletStepSequence)
    return process

def customizeHLTforAlpakaTrackingOnly(process):
    ''' main customization function. Runs tracking only
    in two iterations (initialStep and highPtTripletStep) using patatrack pixel tracks
    '''
    process = customizeHLTforAlpaka(process)
    process = AlpakaHLTTrackingPath(process)
    process.schedule = cms.Schedule(*[
        process.HLT_AlpakaTrackingPath,
        process.endjob_step,
        process.outputmodule
    ])
    return process

def customizeHLTforAlpakaTrackingOnlySingleIteration(process):
    ''' main customization function. Runs tracking only
    in a single iteration (initialStep) using patatrack pixelt tracks
    '''
    process = customizeHLTforAlpakaSingleIteration(process)
    process = AlpakaHLTTrackingPath(process)
    process.schedule = cms.Schedule(*[
        process.HLT_AlpakaTrackingPath,
        process.endjob_step,
        process.outputmodule
    ])
    return process