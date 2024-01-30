import FWCore.ParameterSet.Config as cms


## Pixel HLT in Alpaka
def customizeHLTforDQMGPUvsCPUPixel(process):
    '''Ad-hoc changes to test HLT config containing only DQM_PixelReconstruction_v and DQMGPUvsCPU stream
       only up to the Pixel Local Reconstruction
    '''
    dqmPixelRecoPathName = None
    for pathName in process.paths_():
        if pathName.startswith('DQM_PixelReconstruction_v'):
            dqmPixelRecoPathName = pathName
            break

    if dqmPixelRecoPathName == None:
        return process

    process.hltPixelConsumerGPU.eventProducts = [
        'hltSiPixelClusters',
        'hltSiPixelClustersSoA',
        'hltSiPixelDigis',
        'hltSiPixelRecHits',
        'hltSiPixelRecHitsSoA',
        'hltPixelTracks',
        'hltPixelTracksSoA',
        'hltPixelVertices',
        'hltPixelVerticesSoA',
    ]

    process.hltPixelConsumerCPU.eventProducts = []
    for foo in process.hltPixelConsumerGPU.eventProducts:
        process.hltPixelConsumerCPU.eventProducts += [foo+'CPUSerial']

    # modify EventContent of DQMGPUvsCPU stream
    if hasattr(process, 'hltOutputDQMGPUvsCPU'):
        process.hltOutputDQMGPUvsCPU.outputCommands = [
            'drop *',
            'keep *Cluster*_hltSiPixelClusters_*_*',
            'keep *Cluster*_hltSiPixelClustersLegacyFormatCPUSerial_*_*',
            'keep *_hltSiPixelDigis_*_*',
            'keep *_hltSiPixelDigiErrorsLegacyFormatCPUSerial_*_*',
            'keep *RecHit*_hltSiPixelRecHits_*_*',
            'keep *RecHit*_hltSiPixelRecHitsLegacyFormatCPUSerial_*_*',
            'keep *_hltPixelTracks_*_*',
            'keep *_hltPixelTracksLegacyFormatCPUSerial_*_*',
            'keep *_hltPixelVertices_*_*',
            'keep *_hltPixelVerticesLegacyFormatCPUSerial_*_*',
        ]

    # PixelRecHits: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
    process.hltPixelRecHitsSoAMonitorCPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCPU' )
    )

    # PixelRecHits: monitor of GPU product (Alpaka backend: '')
    process.hltPixelRecHitsSoAMonitorGPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsSoA' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsGPU' )
    )

    # PixelRecHits: 'GPUvsCPU' comparisons
    process.hltPixelRecHitsSoACompareGPUvsCPU = cms.EDProducer('SiPixelPhase1CompareRecHitsSoAAlpaka',
        pixelHitsSrcHost = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        pixelHitsSrcDevice = cms.InputTag( 'hltSiPixelRecHitsSoA' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU' ),
        minD2cut = cms.double( 1.0E-4 )
    )

    process.hltPixelTracksSoAMonitorCPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksCPUSerial'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelTracksSoAMonitorGPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoA'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackGPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelTracksSoACompareGPUvsCPU = cms.EDProducer("SiPixelPhase1CompareTrackSoAAlpaka",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrcHost = cms.InputTag("hltPixelTracksCPUSerial"),
        pixelTrackSrcDevice = cms.InputTag("hltPixelTracksSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelVertexSoAMonitorCPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesCPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCPU')
    )

    process.hltPixelVertexSoAMonitorGPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexGPU')
    )

    process.hltPixelVertexSoACompareGPUvsCPU = cms.EDProducer("SiPixelCompareVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrcHost = cms.InputTag("hltPixelVerticesCPUSerial"),
        pixelVertexSrcDevice = cms.InputTag("hltPixelVerticesSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU')
    )

    process.HLTDQMPixelReconstruction = cms.Sequence(
        process.hltPixelRecHitsSoAMonitorCPU
      + process.hltPixelRecHitsSoAMonitorGPU
      + process.hltPixelRecHitsSoACompareGPUvsCPU
      + process.hltPixelTracksSoAMonitorCPU
      + process.hltPixelTracksSoAMonitorGPU
      + process.hltPixelTracksSoACompareGPUvsCPU
      + process.hltPixelVertexSoAMonitorCPU
      + process.hltPixelVertexSoAMonitorGPU
      + process.hltPixelVertexSoACompareGPUvsCPU
    )

    # Add CPUSerial sequences to DQM_PixelReconstruction_v Path
    dqmPixelRecoPath = getattr(process, dqmPixelRecoPathName)
    try:
        dqmPixelRecoPathIndex = dqmPixelRecoPath.index(process.HLTRecopixelvertexingSequence) + 1
        for cpuSeqName in [
            'HLTDoLocalPixelCPUSerialSequence',
            'HLTRecopixelvertexingCPUSerialSequence',
        ]:
            dqmPixelRecoPath.insert(dqmPixelRecoPathIndex, getattr(process, cpuSeqName))
            dqmPixelRecoPathIndex += 1
    except:
        dqmPixelRecoPathIndex = None

    return process

def customizeHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka
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

    process.hltESPPixelCPEFastParamsPhase1 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase1@alpaka', 
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    ###

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

    # alpaka EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces (* optional)
    #  - SiPixelClustersSoA
    #  - SiPixelDigisSoACollection
    #  - SiPixelDigiErrorsSoACollection *
    #  - SiPixelFormatterErrors *
    process.hltSiPixelClustersSoA = cms.EDProducer('SiPixelRawToClusterPhase1@alpaka',
        mightGet = cms.optional.untracked.vstring,
        IncludeErrors = cms.bool(True),
        UseQualityInfo = cms.bool(False),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        VCaltoElectronGain      = cms.double(1),  # all gains=1, pedestals=0
        VCaltoElectronGain_L1   = cms.double(1),
        VCaltoElectronOffset    = cms.double(0),
        VCaltoElectronOffset_L1 = cms.double(0),
        InputLabel = cms.InputTag('rawDataCollector'),
        Regions = cms.PSet(
            inputs = cms.optional.VInputTag,
            deltaPhi = cms.optional.vdouble,
            maxZ = cms.optional.vdouble,
            beamSpot = cms.optional.InputTag
        ),
        CablingMapLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelClusters = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase1',
        src = cms.InputTag('hltSiPixelClustersSoA'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
    )

    process.hltSiPixelClustersCache = cms.EDProducer('SiPixelClusterShapeCacheProducer',
        src = cms.InputTag( 'hltSiPixelClusters' ),
        onDemand = cms.bool( False )
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
    process.hltSiPixelDigis = cms.EDProducer('SiPixelDigiErrorsFromSoAAlpaka',
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
    process.hltSiPixelRecHitsSoA = cms.EDProducer('SiPixelRecHitAlpakaPhase1@alpaka',
        beamSpot = cms.InputTag('hltOnlineBeamSpotDevice'),
        src = cms.InputTag('hltSiPixelClustersSoA'),
        CPE = cms.string('PixelCPEFastParams'),
        mightGet = cms.optional.untracked.vstring,
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoA'),
        src = cms.InputTag('hltSiPixelClusters'),
    )

    ###
    ### Task: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDevice,
        process.hltSiPixelClustersSoA,
        process.hltSiPixelClusters,   # was: hltSiPixelClusters
        process.hltSiPixelClustersCache,          # really needed ??
        process.hltSiPixelDigis, # was: hltSiPixelDigis
        process.hltSiPixelRecHitsSoA,
        process.hltSiPixelRecHits,    # was: hltSiPixelRecHits
    )

    ###
    ### CPUSerial version of Pixel Local Reconstruction
    ###
    process.hltOnlineBeamSpotDeviceCPUSerial = process.hltOnlineBeamSpotDevice.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersCPUSerial = process.hltSiPixelClustersSoA.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersLegacyFormatCPUSerial = process.hltSiPixelClusters.clone(
        src = 'hltSiPixelClustersCPUSerial'
    )

    process.hltSiPixelDigiErrorsLegacyFormatCPUSerial = process.hltSiPixelDigis.clone(
        digiErrorSoASrc = 'hltSiPixelClustersCPUSerial',
        fmtErrorsSoASrc = 'hltSiPixelClustersCPUSerial',
    )

    process.hltSiPixelRecHitsCPUSerial = process.hltSiPixelRecHitsSoA.clone(
        beamSpot = 'hltOnlineBeamSpotDeviceCPUSerial',
        src = 'hltSiPixelClustersCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelRecHitsLegacyFormatCPUSerial = process.hltSiPixelRecHits.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsCPUSerial',
        src = 'hltSiPixelClustersLegacyFormatCPUSerial',
    )

    process.HLTDoLocalPixelCPUSerialTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDeviceCPUSerial,
        process.hltSiPixelClustersCPUSerial,
        process.hltSiPixelClustersLegacyFormatCPUSerial,
        process.hltSiPixelDigiErrorsLegacyFormatCPUSerial,
        process.hltSiPixelRecHitsCPUSerial,
        process.hltSiPixelRecHitsLegacyFormatCPUSerial,
    )

    process.HLTDoLocalPixelCPUSerialSequence = cms.Sequence( process.HLTDoLocalPixelCPUSerialTask )

    return process

def customizeHLTforAlpakaPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''

    # alpaka EDProducer
    # consumes
    #  - TrackingRecHitsSoACollection<TrackerTraits>
    # produces
    #  - TkSoADevice
    process.hltPixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase1@alpaka',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoA'),
        CPE = cms.string('PixelCPEFastParams'),
        ptmin = cms.double(0.9),
        CAThetaCutBarrel = cms.double(0.002),
        CAThetaCutForward = cms.double(0.003),
        hardCurvCut = cms.double(0.0328407225),
        dcaCutInnerTriplet = cms.double(0.15),
        dcaCutOuterTriplet = cms.double(0.25),
        earlyFishbone = cms.bool(True),
        lateFishbone = cms.bool(False),
        fillStatistics = cms.bool(False),
        minHitsPerNtuplet = cms.uint32(3),
        phiCuts = cms.vint32(
            522, 730, 730, 522, 626,
            626, 522, 522, 626, 626,
            626, 522, 522, 522, 522,
            522, 522, 522, 522
        ),
        maxNumberOfDoublets = cms.uint32(524288),
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
            chi2MaxPt = cms.double(10),
            chi2Coeff = cms.vdouble(0.9, 1.8),
            chi2Scale = cms.double(8),
            tripletMinPt = cms.double(0.5),
            tripletMaxTip = cms.double(0.3),
            tripletMaxZip = cms.double(12),
            quadrupletMinPt = cms.double(0.3),
            quadrupletMaxTip = cms.double(0.5),
            quadrupletMaxZip = cms.double(12)
        ),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltPixelTracksCPUSerial = process.hltPixelTracksSoA.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltPixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase1",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('loose'),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
        trackSrc = cms.InputTag("hltPixelTracksSoA")
    )

    process.hltPixelTracksLegacyFormatCPUSerial = process.hltPixelTracks.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsLegacyFormatCPUSerial"),
        trackSrc = cms.InputTag("hltPixelTracksCPUSerial")
    )

    process.HLTRecoPixelTracksTask = cms.ConditionalTask(
        process.hltPixelTracksSoA,
        process.hltPixelTracks,
    )

    process.HLTRecoPixelTracksCPUSerialTask = cms.ConditionalTask(
        process.hltPixelTracksCPUSerial,
        process.hltPixelTracksLegacyFormatCPUSerial,
    )

    process.HLTRecoPixelTracksCPUSerialSequence = cms.Sequence( process.HLTRecoPixelTracksCPUSerialTask )

    return process

def customizeHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''

    # alpaka EDProducer
    # consumes
    #  - TkSoADevice
    # produces
    #  - ZVertexDevice
    process.hltPixelVerticesSoA = cms.EDProducer('PixelVertexProducerAlpakaPhase1@alpaka',
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

    process.hltPixelVerticesCPUSerial = process.hltPixelVerticesSoA.clone(
        pixelTrackSrc = 'hltPixelTracksCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltPixelVertices = cms.EDProducer("PixelVertexProducerFromSoAAlpaka",
        TrackCollection = cms.InputTag("hltPixelTracks"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltPixelVerticesSoA")
    )

    process.hltPixelVerticesLegacyFormatCPUSerial = process.hltPixelVertices.clone(
        TrackCollection = cms.InputTag("hltPixelTracksLegacyFormatCPUSerial"),
        src = cms.InputTag("hltPixelVerticesCPUSerial")
    )

    process.HLTRecopixelvertexingTask = cms.ConditionalTask(
        process.HLTRecoPixelTracksTask,
        process.hltPixelVerticesSoA,
        process.hltPixelVertices,
        process.hltTrimmedPixelVertices 
    )

    process.HLTRecopixelvertexingCPUSerialTask = cms.ConditionalTask(
        process.HLTRecoPixelTracksCPUSerialTask,
        process.hltPixelVerticesCPUSerial,
        process.hltPixelVerticesLegacyFormatCPUSerial,
    )

    process.HLTRecopixelvertexingCPUSerialSequence = cms.Sequence( process.HLTRecopixelvertexingCPUSerialTask )

    return process

def customizeHLTforAlpakaPixelRecoTheRest(process):
    '''Customize HLT path depending on old SoA tracks
    '''
    process.hltL2TauTagNNProducer = cms.EDProducer("L2TauNNProducerAlpaka",
        BeamSpot = cms.InputTag("hltOnlineBeamSpot"),
        L1Taus = cms.VPSet(
            cms.PSet(
                L1CollectionName = cms.string('DoubleTau'),
                L1TauTrigger = cms.InputTag("hltL1sDoubleTauBigOR")
            ),
            cms.PSet(
                L1CollectionName = cms.string('SingleTau'),
                L1TauTrigger = cms.InputTag("hltL1sSingleTau")
            ),
            cms.PSet(
                L1CollectionName = cms.string('MuXXTauYY'),
                L1TauTrigger = cms.InputTag("hltL1sBigOrMuXXerIsoTauYYer")
            ),
            cms.PSet(
                L1CollectionName = cms.string('Mu22Tau40'),
                L1TauTrigger = cms.InputTag("hltL1sMu22erIsoTau40er")
            ),
            cms.PSet(
                L1CollectionName = cms.string('DoubleTauJet'),
                L1TauTrigger = cms.InputTag("hltL1sBigORDoubleTauJet")
            ),
            cms.PSet(
                L1CollectionName = cms.string('VBFIsoTau'),
                L1TauTrigger = cms.InputTag("hltL1VBFDiJetIsoTau")
            ),
            cms.PSet(
                L1CollectionName = cms.string('Mu18TauXX'),
                L1TauTrigger = cms.InputTag("hltL1sVeryBigORMu18erTauXXer2p1")
            ),
            cms.PSet(
                L1CollectionName = cms.string('DoubleTauLowMass'),
                L1TauTrigger = cms.InputTag("hltL1sDoubleTauBigORWithLowMass")
            )
        ),
        debugLevel = cms.int32(0),
        ebInput = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
        eeInput = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
        fractionSumPt2 = cms.double(0.3),
        graphPath = cms.string('RecoTauTag/TrainingFiles/data/L2TauNNTag/L2TauTag_Run3v1.pb'),
        hbheInput = cms.InputTag("hltHbhereco"),
        hoInput = cms.InputTag("hltHoreco"),
        maxVtx = cms.uint32(100),
        minSumPt2 = cms.double(0.0),
        normalizationDict = cms.string('RecoTauTag/TrainingFiles/data/L2TauNNTag/NormalizationDict.json'),
        pataTracks = cms.InputTag("hltPixelTracksSoA"),
        pataVertices = cms.InputTag("hltPixelVerticesSoA"),
        track_chi2_max = cms.double(99999.0),
        track_pt_max = cms.double(10.0),
        track_pt_min = cms.double(1.0)
    )
    
    return process

def customizeHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''

    process = customizeHLTforAlpakaPixelRecoLocal(process)
    process = customizeHLTforAlpakaPixelRecoTracking(process)
    process = customizeHLTforAlpakaPixelRecoVertexing(process)
    process = customizeHLTforDQMGPUvsCPUPixel(process)    
    process = customizeHLTforAlpakaPixelRecoTheRest(process)

    return process

## ECAL HLT in Alpaka

def customizeHLTforAlpakaEcalLocalReco(process):
    
    if hasattr(process, 'hltEcalDigisGPU'):
        process.hltEcalDigisPortable = cms.EDProducer("EcalRawToDigiPortable@alpaka",
            FEDs = process.hltEcalDigisGPU.FEDs,
            InputLabel = process.hltEcalDigisGPU.InputLabel,
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
            ),
            digisLabelEB = process.hltEcalDigisGPU.digisLabelEB,
            digisLabelEE = process.hltEcalDigisGPU.digisLabelEE,
            maxChannelsEB = process.hltEcalDigisGPU.maxChannelsEB,
            maxChannelsEE = process.hltEcalDigisGPU.maxChannelsEE,
            mightGet = cms.optional.untracked.vstring
        )
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.hltEcalDigisPortable)

        process.load("EventFilter.EcalRawToDigi.ecalElectronicsMappingHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalElectronicsMappingHostESProducer)

        delattr(process, 'hltEcalDigisGPU')
        delattr(process, 'ecalElectronicsMappingGPUESProducer')

    if hasattr(process, 'hltEcalDigisFromGPU'):
        process.hltEcalDigisFromGPU = cms.EDProducer( "EcalDigisFromPortableProducer",
            digisInLabelEB = cms.InputTag( 'hltEcalDigisPortable','ebDigis' ),
            digisInLabelEE = cms.InputTag( 'hltEcalDigisPortable','eeDigis' ),
            digisOutLabelEB = cms.string( "ebDigis" ),
            digisOutLabelEE = cms.string( "eeDigis" ),
            produceDummyIntegrityCollections = cms.bool( False )
        )

    if hasattr(process, 'hltEcalUncalibRecHitGPU'):
        process.hltEcalUncalibRecHitPortable = cms.EDProducer("EcalUncalibRecHitProducerPortable@alpaka",
            EBtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EBtimeConstantTerm,
            EBtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Lower,
            EBtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Upper,
            EBtimeNconst = process.hltEcalUncalibRecHitGPU.EBtimeNconst,
            EEtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EEtimeConstantTerm,
            EEtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Lower,
            EEtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Upper,
            EEtimeNconst = process.hltEcalUncalibRecHitGPU.EEtimeNconst,
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
            ),
            amplitudeThresholdEB = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEB,
            amplitudeThresholdEE = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEE,
            digisLabelEB = cms.InputTag("hltEcalDigisPortable","ebDigis"),
            digisLabelEE = cms.InputTag("hltEcalDigisPortable","eeDigis"),
            kernelMinimizeThreads = process.hltEcalUncalibRecHitGPU.kernelMinimizeThreads,
            mightGet = cms.optional.untracked.vstring,
            outOfTimeThresholdGain12mEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12mEB,
            outOfTimeThresholdGain12mEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12mEE,
            outOfTimeThresholdGain12pEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12pEB,
            outOfTimeThresholdGain12pEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12pEE,
            outOfTimeThresholdGain61mEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61mEB,
            outOfTimeThresholdGain61mEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61mEE,
            outOfTimeThresholdGain61pEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61pEB,
            outOfTimeThresholdGain61pEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61pEE,
            recHitsLabelEB = process.hltEcalUncalibRecHitGPU.recHitsLabelEB,
            recHitsLabelEE = process.hltEcalUncalibRecHitGPU.recHitsLabelEE,
            shouldRunTimingComputation = process.hltEcalUncalibRecHitGPU.shouldRunTimingComputation
        )
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.hltEcalUncalibRecHitPortable)

        process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalMultifitConditionsHostESProducer)

        process.ecalMultifitParametersSource = cms.ESSource("EmptyESSource",
            firstValid = cms.vuint32(1),
            iovIsRunNotTime = cms.bool(True),
            recordName = cms.string('EcalMultifitParametersRcd')
        )
        process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitParametersHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalMultifitParametersHostESProducer)

        delattr(process, 'hltEcalUncalibRecHitGPU')

        if hasattr(process, 'hltEcalUncalibRecHitFromSoA'):
            process.hltEcalUncalibRecHitFromSoA = cms.EDProducer("EcalUncalibRecHitSoAToLegacy",
                isPhase2 = process.hltEcalUncalibRecHitFromSoA.isPhase2,
                mightGet = cms.optional.untracked.vstring,
                recHitsLabelCPUEB = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEB,
                recHitsLabelCPUEE = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEE,
                uncalibRecHitsPortableEB = cms.InputTag("hltEcalUncalibRecHitPortable","EcalUncalibRecHitsEB"),
                uncalibRecHitsPortableEE = cms.InputTag("hltEcalUncalibRecHitPortable","EcalUncalibRecHitsEE")
            )

        if hasattr(process, 'hltEcalUncalibRecHitSoA'):
            delattr(process, 'hltEcalUncalibRecHitSoA')

    process.HLTDoFullUnpackingEgammaEcalTask = cms.ConditionalTask(process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask, process.HLTPreshowerTask)

    return process

def customizeHLTforAlpaka(process):

    process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
    process.load('Configuration.StandardSequences.Accelerators_cff')
    
    process = customizeHLTforAlpakaEcalLocalReco(process)
    process = customizeHLTforAlpakaPixelReco(process)

    return process

