import FWCore.ParameterSet.Config as cms

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
        'hltSiPixelClustersLegacyFormat',
        'hltSiPixelDigiErrorsLegacyFormat',
        'hltSiPixelRecHits',
        'hltSiPixelRecHitsLegacyFormat',
        'hltPixelTracks',
        'hltPixelTracksLegacyFormat',
        'hltPixelVertices',
        'hltPixelVerticesLegacyFormat',
    ]

    process.hltPixelConsumerCPU.eventProducts = []
    for foo in process.hltPixelConsumerGPU.eventProducts:
        process.hltPixelConsumerCPU.eventProducts += [foo+'CPUSerial']

    # modify EventContent of DQMGPUvsCPU stream
    if hasattr(process, 'hltOutputDQMGPUvsCPU'):
        process.hltOutputDQMGPUvsCPU.outputCommands = [
            'drop *',
            'keep *Cluster*_hltSiPixelClustersLegacyFormat_*_*',
            'keep *Cluster*_hltSiPixelClustersLegacyFormatCPUSerial_*_*',
            'keep *_hltSiPixelDigiErrorsLegacyFormat_*_*',
            'keep *_hltSiPixelDigiErrorsLegacyFormatCPUSerial_*_*',
            'keep *RecHit*_hltSiPixelRecHitsLegacyFormat_*_*',
            'keep *RecHit*_hltSiPixelRecHitsLegacyFormatCPUSerial_*_*',
            'keep *_hltPixelTracksLegacyFormat_*_*',
            'keep *_hltPixelTracksLegacyFormatCPUSerial_*_*',
            'keep *_hltPixelVerticesLegacyFormat_*_*',
            'keep *_hltPixelVerticesLegacyFormatCPUSerial_*_*',
        ]

    # PixelRecHits: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
    process.hltSiPixelRecHitsSoAMonitorCPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCPU' )
    )

    # PixelRecHits: monitor of GPU product (Alpaka backend: '')
    process.hltSiPixelRecHitsSoAMonitorGPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHits' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsGPU' )
    )

    # PixelRecHits: 'GPUvsCPU' comparisons
    process.hltSiPixelRecHitsSoACompareGPUvsCPU = cms.EDProducer('SiPixelPhase1CompareRecHitsSoAAlpaka',
        pixelHitsSrcCPU = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        pixelHitsSrcGPU = cms.InputTag( 'hltSiPixelRecHits' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU' ),
        minD2cut = cms.double( 1.0E-4 )
    )

    process.hltSiPixelTrackSoAMonitorCPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksCPUSerial'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltSiPixelTrackSoAMonitorGPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracks'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackGPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltSiPixelTrackSoACompareGPUvsCPU = cms.EDProducer("SiPixelPhase1CompareTrackSoAAlpaka",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrcCPU = cms.InputTag("hltPixelTracksCPUSerial"),
        pixelTrackSrcGPU = cms.InputTag("hltPixelTracks"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltSiPixelVertexSoAMonitorCPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesCPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCPU')
    )

    process.hltSiPixelVertexSoAMonitorGPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVertices"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexGPU')
    )

    process.hltSiPixelVertexSoACompareGPUvsCPU = cms.EDProducer("SiPixelCompareVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrcCPU = cms.InputTag("hltPixelVerticesCPUSerial"),
        pixelVertexSrcGPU = cms.InputTag("hltPixelVertices"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU')
    )

    process.HLTDQMPixelReconstruction = cms.Sequence(
        process.hltSiPixelRecHitsSoAMonitorCPU
      + process.hltSiPixelRecHitsSoAMonitorGPU
      + process.hltSiPixelRecHitsSoACompareGPUvsCPU
      + process.hltSiPixelTrackSoAMonitorCPU
      + process.hltSiPixelTrackSoAMonitorGPU
      + process.hltSiPixelTrackSoACompareGPUvsCPU
      + process.hltSiPixelVertexSoAMonitorCPU
      + process.hltSiPixelVertexSoAMonitorGPU
      + process.hltSiPixelVertexSoACompareGPUvsCPU
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
    #  - BeamSpotDeviceProduct
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
    process.hltSiPixelClusters = cms.EDProducer('SiPixelRawToClusterPhase1@alpaka',
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

    process.hltSiPixelClustersLegacyFormat = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase1',
        src = cms.InputTag('hltSiPixelClusters'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
    )

    process.hltSiPixelClustersCache = cms.EDProducer('SiPixelClusterShapeCacheProducer',
        src = cms.InputTag( 'hltSiPixelClustersLegacyFormat' ),
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
    process.hltSiPixelDigiErrorsLegacyFormat = cms.EDProducer('SiPixelDigiErrorsFromSoAAlpaka',
        digiErrorSoASrc = cms.InputTag('hltSiPixelClusters'),
        fmtErrorsSoASrc = cms.InputTag('hltSiPixelClusters'),
        CablingMapLabel = cms.string(''),
        UsePhase1 = cms.bool(True),
        ErrorList = cms.vint32(29),
        UserErrorList = cms.vint32(40)
    )

    # alpaka EDProducer
    # consumes
    #  - BeamSpotDeviceProduct
    #  - SiPixelClustersSoA
    #  - SiPixelDigisCollection
    # produces
    #  - TrackingRecHitAlpakaCollection<TrackerTraits>
    process.hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitAlpakaPhase1@alpaka',
        beamSpot = cms.InputTag('hltOnlineBeamSpotDevice'),
        src = cms.InputTag('hltSiPixelClusters'),
        CPE = cms.string('PixelCPEFastParams'),
        mightGet = cms.optional.untracked.vstring,
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelRecHitsLegacyFormat = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHits'),
        src = cms.InputTag('hltSiPixelClustersLegacyFormat'),
    )

    ###
    ### Task: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotDevice,
        process.hltSiPixelClusters,
        process.hltSiPixelClustersLegacyFormat,   # was: hltSiPixelClusters
        process.hltSiPixelClustersCache,          # really needed ??
        process.hltSiPixelDigiErrorsLegacyFormat, # was: hltSiPixelDigis
        process.hltSiPixelRecHits,
        process.hltSiPixelRecHitsLegacyFormat,    # was: hltSiPixelRecHits
    )

    ###
    ### CPUSerial version of Pixel Local Reconstruction
    ###
    process.hltOnlineBeamSpotDeviceCPUSerial = process.hltOnlineBeamSpotDevice.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersCPUSerial = process.hltSiPixelClusters.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelClustersLegacyFormatCPUSerial = process.hltSiPixelClustersLegacyFormat.clone(
        src = 'hltSiPixelClustersCPUSerial'
    )

    process.hltSiPixelDigiErrorsLegacyFormatCPUSerial = process.hltSiPixelDigiErrorsLegacyFormat.clone(
        digiErrorSoASrc = 'hltSiPixelClustersCPUSerial',
        fmtErrorsSoASrc = 'hltSiPixelClustersCPUSerial',
    )

    process.hltSiPixelRecHitsCPUSerial = process.hltSiPixelRecHits.clone(
        beamSpot = 'hltOnlineBeamSpotDeviceCPUSerial',
        src = 'hltSiPixelClustersCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltSiPixelRecHitsLegacyFormatCPUSerial = process.hltSiPixelRecHitsLegacyFormat.clone(
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
    process.hltPixelTracks = cms.EDProducer('CAHitNtupletAlpakaPhase1@alpaka',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHits'),
        CPE = cms.string('PixelCPEFastParams'),
        ptmin = cms.double(0.89999997615814209),
        CAThetaCutBarrel = cms.double(0.0020000000949949026),
        CAThetaCutForward = cms.double(0.0030000000260770321),
        hardCurvCut = cms.double(0.032840722495894911),
        dcaCutInnerTriplet = cms.double(0.15000000596046448),
        dcaCutOuterTriplet = cms.double(0.25),
        earlyFishbone = cms.bool(True),
        lateFishbone = cms.bool(False),
        fillStatistics = cms.bool(False),
        minHitsPerNtuplet = cms.uint32(3),
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

    process.hltPixelTracksCPUSerial = process.hltPixelTracks.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltPixelTracksLegacyFormat = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase1",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('loose'),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsLegacyFormat"),
        trackSrc = cms.InputTag("hltPixelTracks")
    )

    process.hltPixelTracksLegacyFormatCPUSerial = process.hltPixelTracksLegacyFormat.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsLegacyFormatCPUSerial"),
        trackSrc = cms.InputTag("hltPixelTracksCPUSerial")
    )

    process.HLTRecoPixelTracksTask = cms.ConditionalTask(
        process.hltPixelTracks,
        process.hltPixelTracksLegacyFormat,
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
    process.hltPixelVertices = cms.EDProducer('PixelVertexProducerAlpakaPhase1@alpaka',
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
        pixelTrackSrc = cms.InputTag('hltPixelTracks'),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltPixelVerticesCPUSerial = process.hltPixelVertices.clone(
        pixelTrackSrc = 'hltPixelTracksCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

    process.hltPixelVerticesLegacyFormat = cms.EDProducer("PixelVertexProducerFromSoAAlpaka",
        TrackCollection = cms.InputTag("hltPixelTracksLegacyFormat"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltPixelVertices")
    )

    process.hltPixelVerticesLegacyFormatCPUSerial = process.hltPixelVerticesLegacyFormat.clone(
        TrackCollection = cms.InputTag("hltPixelTracksLegacyFormatCPUSerial"),
        src = cms.InputTag("hltPixelVerticesCPUSerial")
    )

    process.HLTRecopixelvertexingTask = cms.ConditionalTask(
        process.HLTRecoPixelTracksTask,
        process.hltPixelVertices,
        process.hltPixelVerticesLegacyFormat,
    )

    process.HLTRecopixelvertexingCPUSerialTask = cms.ConditionalTask(
        process.HLTRecoPixelTracksCPUSerialTask,
        process.hltPixelVerticesCPUSerial,
        process.hltPixelVerticesLegacyFormatCPUSerial,
    )

    process.HLTRecopixelvertexingCPUSerialSequence = cms.Sequence( process.HLTRecopixelvertexingCPUSerialTask )

    return process

def customizeHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''
    process.load('Configuration.StandardSequences.Accelerators_cff')
    process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

    process = customizeHLTforAlpakaPixelRecoLocal(process)
    process = customizeHLTforAlpakaPixelRecoTracking(process)
    process = customizeHLTforAlpakaPixelRecoVertexing(process)

    return process

def customizeHLTforPatatrack(process):
    '''Customize HLT configuration introducing latest Patatrack developments
    '''
    process = customizeHLTforAlpakaPixelReco(process)
    return process
