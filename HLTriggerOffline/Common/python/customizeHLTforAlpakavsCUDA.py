import FWCore.ParameterSet.Config as cms
import copy

## Pixel HLT in Alpaka
def customizeHLTforDQMAlpakavsCUDAPixel(process):
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
        'hltSiPixelClustersCUDA',
        'hltSiPixelClustersSoACUDA',
        'hltSiPixelDigisCUDA',
        'hltSiPixelDigisSoACUDA',
        'hltSiPixelRecHitsCUDA',
        'hltSiPixelRecHitsSoACUDA',
        'hltPixelTracksCUDA',
        'hltPixelTracksSoACUDA',
        'hltPixelVerticesCUDA',
        'hltPixelVerticesSoACUDA',
    ]

    process.hltPixelConsumerCPU.eventProducts = []
    for foo in process.hltPixelConsumerGPU.eventProducts:
        process.hltPixelConsumerCPU.eventProducts += [foo+'CPUSerial']
    # Left over legacy format collections
    process.hltPixelConsumerCPU.eventProducts += ['hltPixelTracksLegacyFormatCPUSerial',
                                                  'hltPixelTracksLegacyFormatCUDACPUSerial',
                                                  'hltPixelVerticesLegacyFormatCPUSerial',
                                                  'hltPixelVerticesLegacyFormatCUDACPUSerial',
                                                  'hltSiPixelClustersLegacyFormatCPUSerial',
                                                  'hltSiPixelDigiErrorsLegacyFormatCPUSerial',
                                                  'hltSiPixelRecHitsLegacyFormatCPUSerial',
                                                  'hltSiPixelRecHitsLegacyFormatCUDACPUSerial',
                                                  'hltSiPixelDigisLegacyFormatCUDACPUSerial']

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
            'keep *Cluster*_hltSiPixelClustersCUDA_*_*',
            'keep *Cluster*_hltSiPixelClustersLegacyFormatCUDACPUSerial_*_*',
            'keep *_hltSiPixelDigisCUDA_*_*',
            'keep *_hltSiPixelDigisLegacyFormatCUDACPUSerial_*_*',
            'keep *RecHit*_hltSiPixelRecHitsCUDA_*_*',
            'keep *RecHit*_hltSiPixelRecHitsLegacyFormatCUDACPUSerial_*_*',
            'keep *_hltPixelTracksCUDA_*_*',
            'keep *_hltPixelTracksLegacyFormatCUDACPUSerial_*_*',
            'keep *_hltPixelVerticesCUDA_*_*',
            'keep *_hltPixelVerticesLegacyFormatCUDACPUSerial_*_*',
        ]

    from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import SiPixelPhase1RawDataConf,SiPixelPhase1RawDataAnalyzer

    # PixelDigiErrors: monitor of CPUSerial product
    SiPixelPhase1RawDataConfForSerial = copy.deepcopy(SiPixelPhase1RawDataConf)
    for pset in SiPixelPhase1RawDataConfForSerial:
        pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsSerial"

    process.hltPixelPhase1MonitorRawDataASerial = SiPixelPhase1RawDataAnalyzer.clone(
        src = "hltSiPixelDigiErrorsLegacyFormatCPUSerial",
        histograms = SiPixelPhase1RawDataConfForSerial
    )

    # PixelDigiErrors: monitor of GPU product
    SiPixelPhase1RawDataConfForDevice = copy.deepcopy(SiPixelPhase1RawDataConf)
    for pset in SiPixelPhase1RawDataConfForDevice:
        pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsDevice"

    process.hltPixelPhase1MonitorRawDataADevice = SiPixelPhase1RawDataAnalyzer.clone(
        src = "hltSiPixelDigis",
        histograms = SiPixelPhase1RawDataConfForDevice
    )

    # PixelDigiErrors: monitor of CPUSerial product with CUDA
    SiPixelPhase1RawDataConfForCPU = copy.deepcopy(SiPixelPhase1RawDataConf)
    for pset in SiPixelPhase1RawDataConfForCPU:
        pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsCPU"

    process.hltPixelPhase1MonitorRawDataACPU = SiPixelPhase1RawDataAnalyzer.clone(
        src = "hltSiPixelDigisLegacyFormatCUDACPUSerial",
        histograms = SiPixelPhase1RawDataConfForCPU
    )

    # PixelDigiErrors: monitor of GPU product with CUDA
    SiPixelPhase1RawDataConfForGPU = copy.deepcopy(SiPixelPhase1RawDataConf)
    for pset in SiPixelPhase1RawDataConfForGPU:
        pset.topFolderName =  "SiPixelHeterogeneous/PixelErrorsGPU"

    process.hltPixelPhase1MonitorRawDataAGPU = SiPixelPhase1RawDataAnalyzer.clone(
        src = "hltSiPixelDigisCUDA",
        histograms = SiPixelPhase1RawDataConfForGPU
    )

    # PixelDigiErrors: 'Alpaka' comparison
    process.hltPixelDigiErrorsCompareAlpaka = cms.EDProducer('SiPixelPhase1RawDataErrorComparator',
        pixelErrorSrcCPU = cms.InputTag( 'hltSiPixelDigiErrorsLegacyFormatCPUSerial' ),
        pixelErrorSrcGPU = cms.InputTag( 'hltSiPixelDigis' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelErrorsCompareAlpaka' )
    )

    # PixelDigiErrors: 'CUDA' comparison
    process.hltPixelDigiErrorsCompareCUDA = cms.EDProducer('SiPixelPhase1RawDataErrorComparator',
        pixelErrorSrcCPU = cms.InputTag( 'hltSiPixelDigisLegacyFormatCUDACPUSerial' ),
        pixelErrorSrcGPU = cms.InputTag( 'hltSiPixelDigisCUDA' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelErrorsCompareCUDA' )
    )

    # PixelDigiErrors: 'AlpakavsCUDA' comparisons
    process.hltPixelDigiErrorsCompareAlpakavsCUDACPU = cms.EDProducer('SiPixelPhase1RawDataErrorComparator',
        pixelErrorSrcCPU = cms.InputTag( 'hltSiPixelDigiErrorsLegacyFormatCPUSerial' ), # Alpaka Serial
        pixelErrorSrcGPU = cms.InputTag( 'hltSiPixelDigisLegacyFormatCUDACPUSerial' ),              # CUDA CPU
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelErrorsCompareAlpakavsCUDACPU' )
    )

    process.hltPixelDigiErrorsCompareAlpakavsCUDAGPU = cms.EDProducer('SiPixelPhase1RawDataErrorComparator',
        pixelErrorSrcCPU = cms.InputTag( 'hltSiPixelDigis' ),       # Alpaka Device
        pixelErrorSrcGPU = cms.InputTag( 'hltSiPixelDigisCUDA' ),   # CUDA GPU
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelErrorsCompareAlpakavsCUDAGPU' )
    )

    # PixelRecHits: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
    process.hltPixelRecHitsSoAMonitorSerial = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsSerial' )
    )

    # PixelRecHits: monitor of GPU product (Alpaka backend: '')
    process.hltPixelRecHitsSoAMonitorDevice = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsSoA' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsDevice' )
    )

    # PixelRecHits: monitor of CPUSerial product with CUDA
    process.hltPixelRecHitsSoAMonitorCPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoA',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsSoACUDACPUSerial' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCPU' )
    )

    # PixelRecHits: monitor of GPU product with CUDA
    process.hltPixelRecHitsSoAMonitorGPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoA',
        pixelHitsSrc = cms.InputTag( 'hltSiPixelRecHitsSoAFromCUDA' ),
        TopFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsGPU' )
    )

    # PixelRecHits: 'Alpaka' comparison
    process.hltPixelRecHitsSoACompareAlpaka = cms.EDProducer('SiPixelPhase1CompareRecHits',
        pixelHitsReferenceSoA = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        pixelHitsTargetSoA = cms.InputTag( 'hltSiPixelRecHitsSoA' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareAlpaka' ),
        minD2cut = cms.double( 1.0E-4 ),
        case = cms.string('Alpaka')
    )

    # PixelRecHits: 'CUDA' comparison
    process.hltPixelRecHitsSoACompareCUDA = cms.EDProducer('SiPixelPhase1CompareRecHits',
        pixelHitsReferenceCUDA = cms.InputTag( 'hltSiPixelRecHitsSoACUDACPUSerial' ),
        pixelHitsTargetCUDA = cms.InputTag( 'hltSiPixelRecHitsSoAFromCUDA' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareCUDA' ),
        minD2cut = cms.double( 1.0E-4 ),
        case = cms.string('CUDA')
    )

    # PixelRecHits: 'AlpakavsCUDA' comparisons
    process.hltPixelRecHitsSoACompareAlpakavsCUDACPU = cms.EDProducer('SiPixelPhase1CompareRecHits',
        pixelHitsReferenceSoA = cms.InputTag( 'hltSiPixelRecHitsCPUSerial' ),
        pixelHitsTargetCUDA = cms.InputTag( 'hltSiPixelRecHitsSoACUDACPUSerial' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareAlpakavsCUDACPU' ),
        minD2cut = cms.double( 1.0E-4 ),
        case = cms.string('AlpakavsCUDA')
    )

    process.hltPixelRecHitsSoACompareAlpakavsCUDAGPU = cms.EDProducer('SiPixelPhase1CompareRecHits',
        pixelHitsReferenceSoA = cms.InputTag( 'hltSiPixelRecHitsSoA' ),
        pixelHitsTargetCUDA = cms.InputTag( 'hltSiPixelRecHitsSoAFromCUDA' ),
        topFolderName = cms.string( 'SiPixelHeterogeneous/PixelRecHitsCompareAlpakavsCUDAGPU' ),
        minD2cut = cms.double( 1.0E-4 ),
        case = cms.string('AlpakavsCUDA')
    )

    # PixelTracks: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
    process.hltPixelTracksSoAMonitorSerial = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksCPUSerial'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackSerial'),
        useQualityCut = cms.bool(True)
    )

    # PixelTracks: monitor of GPU product (Alpaka backend: '')
    process.hltPixelTracksSoAMonitorDevice = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoA'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackDevice'),
        useQualityCut = cms.bool(True)
    )

    # PixelTracks: monitor of CPUSerial product with CUDA
    process.hltPixelTracksSoAMonitorCPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoA",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoACUDACPUSerial'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCPU'),
        useQualityCut = cms.bool(True)
    )

    # PixelTracks: monitor of GPU product with CUDA
    process.hltPixelTracksSoAMonitorGPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoA",
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoAFromCUDA'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackGPU'),
        useQualityCut = cms.bool(True)
    )

    # PixelTracks: 'Alpaka' comparison
    process.hltPixelTracksSoACompareAlpaka = cms.EDProducer("SiPixelPhase1CompareTracks",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackReferenceSoA = cms.InputTag("hltPixelTracksCPUSerial"),
        pixelTrackTargetSoA = cms.InputTag("hltPixelTracksSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareAlpaka'),
        useQualityCut = cms.bool(True),
        case = cms.string('Alpaka')
    )

    # PixelTracks: 'CUDA' comparison
    process.hltPixelTracksSoACompareCUDA = cms.EDProducer("SiPixelPhase1CompareTracks",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackReferenceCUDA = cms.InputTag("hltPixelTracksSoACUDACPUSerial"),
        pixelTrackTargetCUDA = cms.InputTag("hltPixelTracksSoAFromCUDA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareCUDA'),
        useQualityCut = cms.bool(True),
        case = cms.string('CUDA')
    )

    # PixelTracks: 'AlpakavsCUDA' comparisons
    process.hltPixelTracksSoACompareAlpakavsCUDACPU = cms.EDProducer("SiPixelPhase1CompareTracks",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackReferenceSoA = cms.InputTag("hltPixelTracksCPUSerial"),
        pixelTrackTargetCUDA = cms.InputTag("hltPixelTracksSoACUDACPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareAlpakavsCUDACPU'),
        useQualityCut = cms.bool(True),
        case = cms.string('AlpakavsCUDA')
    )

    process.hltPixelTracksSoACompareAlpakavsCUDAGPU = cms.EDProducer("SiPixelPhase1CompareTracks",
        deltaR2cut = cms.double(0.04),
        mightGet = cms.optional.untracked.vstring,
        minQuality = cms.string('loose'),
        pixelTrackReferenceSoA = cms.InputTag("hltPixelTracksSoA"),
        pixelTrackTargetCUDA = cms.InputTag("hltPixelTracksSoAFromCUDA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareAlpakavsCUDAGPU'),
        useQualityCut = cms.bool(True),
        case = cms.string('AlpakavsCUDA')
    )

    # PixelVertices: monitor of CPUSerial product (Alpaka backend: 'serial_sync')
    process.hltPixelVertexSoAMonitorSerial = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesCPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexSerial')
    )

    # PixelVertices: monitor of GPU product (Alpaka backend: '')
    process.hltPixelVertexSoAMonitorDevice = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexDevice')
    )

    # PixelTracks: monitor of CPUSerial product with CUDA
    process.hltPixelVertexSoAMonitorCPU = cms.EDProducer("SiPixelMonitorVertexSoA",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoACUDACPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCPU')
    )

    # PixelTracks: monitor of GPU product with CUDA
    process.hltPixelVertexSoAMonitorGPU = cms.EDProducer("SiPixelMonitorVertexSoA",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoAFromCUDA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexGPU')
    )

    # PixelTracks: 'Alpaka' comparison
    process.hltPixelVertexSoACompareAlpaka = cms.EDProducer("SiPixelCompareVertices",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexReferenceSoA = cms.InputTag("hltPixelVerticesCPUSerial"),
        pixelVertexTargetSoA = cms.InputTag("hltPixelVerticesSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareAlpaka'),
        case = cms.string('Alpaka')
    )

    # PixelTracks: 'CUDA' comparison
    process.hltPixelVertexSoACompareCUDA = cms.EDProducer("SiPixelCompareVertices",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexReferenceCUDA = cms.InputTag("hltPixelVerticesSoACUDACPUSerial"),
        pixelVertexTargetCUDA = cms.InputTag("hltPixelVerticesSoAFromCUDA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareCUDA'),
        case = cms.string('CUDA')
    )

    # PixelTracks: 'AlpakavsCUDA' comparisons
    process.hltPixelVertexSoACompareAlpakavsCUDACPU = cms.EDProducer("SiPixelCompareVertices",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexReferenceSoA = cms.InputTag("hltPixelVerticesCPUSerial"),
        pixelVertexTargetCUDA = cms.InputTag("hltPixelVerticesSoACUDACPUSerial"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareAlpakavsCUDACPU'),
        case = cms.string('AlpakavsCUDA')
    )

    process.hltPixelVertexSoACompareAlpakavsCUDAGPU = cms.EDProducer("SiPixelCompareVertices",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        mightGet = cms.optional.untracked.vstring,
        pixelVertexReferenceSoA = cms.InputTag("hltPixelVerticesSoA"),
        pixelVertexTargetCUDA = cms.InputTag("hltPixelVerticesSoAFromCUDA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCompareAlpakavsCUDAGPU'),
        case = cms.string('AlpakavsCUDA')
    )

    process.HLTDQMPixelReconstruction = cms.Sequence(
        process.hltPixelPhase1MonitorRawDataASerial
      + process.hltPixelPhase1MonitorRawDataADevice
      + process.hltPixelPhase1MonitorRawDataACPU
      + process.hltPixelPhase1MonitorRawDataAGPU
      + process.hltPixelDigiErrorsCompareAlpaka
      + process.hltPixelDigiErrorsCompareCUDA
      + process.hltPixelDigiErrorsCompareAlpakavsCUDACPU
      + process.hltPixelDigiErrorsCompareAlpakavsCUDAGPU
      + process.hltPixelRecHitsSoAMonitorSerial
      + process.hltPixelRecHitsSoAMonitorDevice
      + process.hltPixelRecHitsSoAMonitorCPU
      + process.hltPixelRecHitsSoAMonitorGPU
      + process.hltPixelRecHitsSoACompareAlpaka
      + process.hltPixelRecHitsSoACompareCUDA
      + process.hltPixelRecHitsSoACompareAlpakavsCUDACPU
      + process.hltPixelRecHitsSoACompareAlpakavsCUDAGPU
      + process.hltPixelTracksSoAMonitorSerial
      + process.hltPixelTracksSoAMonitorDevice
      + process.hltPixelTracksSoAMonitorCPU
      + process.hltPixelTracksSoAMonitorGPU
      + process.hltPixelTracksSoACompareAlpaka
      + process.hltPixelTracksSoACompareCUDA
      + process.hltPixelTracksSoACompareAlpakavsCUDACPU
      + process.hltPixelTracksSoACompareAlpakavsCUDAGPU
      + process.hltPixelVertexSoAMonitorSerial
      + process.hltPixelVertexSoAMonitorDevice
      + process.hltPixelVertexSoAMonitorCPU
      + process.hltPixelVertexSoAMonitorGPU
      + process.hltPixelVertexSoACompareAlpaka
      + process.hltPixelVertexSoACompareCUDA
      + process.hltPixelVertexSoACompareAlpakavsCUDACPU
      + process.hltPixelVertexSoACompareAlpakavsCUDAGPU
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
        for cudaSeqName in [
            'HLTDoLocalPixelCUDASequence',
            'HLTRecopixelvertexingCUDASequence',
            'HLTDoLocalPixelCUDACPUSerialSequence',
            'HLTRecopixelvertexingCUDACPUSerialSequence',
        ]:
            dqmPixelRecoPath.insert(dqmPixelRecoPathIndex, getattr(process, cudaSeqName))
            dqmPixelRecoPathIndex += 1
    except:
        dqmPixelRecoPathIndex = None

    return process

def customizeHLTforCUDAPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in CUDA
    '''

    # Producing records for CUDA
    process.siPixelGainCalibrationForHLTGPU = cms.ESProducer( "SiPixelGainCalibrationForHLTGPUESProducer",
        appendToDataLabel = cms.string( "" )
    )

    process.siPixelROCsStatusAndMappingWrapperESProducer = cms.ESProducer( "SiPixelROCsStatusAndMappingWrapperESProducer",
        ComponentName = cms.string( "" ),
        CablingMapLabel = cms.string( "" ),
        UseQualityInfo = cms.bool( False ),
        appendToDataLabel = cms.string( "" )
    )

    process.pixelCPEFastESProducerPhase1 = cms.ESProducer("PixelCPEFastESProducerPhase1",
        appendToDataLabel = cms.string(''),
    )

    ###

    # CUDA EDProducer
    # consumes
    #  - reco::BeamSpot
    # produces
    #  - BeamSpotCUDA
    process.hltOnlineBeamSpotGPU = cms.EDProducer("BeamSpotToCUDA",
        src = cms.InputTag("hltOnlineBeamSpot")
    )

    # CUDA EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces (* optional)
    #  - SiPixelDigisCUDA
    #  - SiPixelDigiErrorsCUDA
    #  - SiPixelClustersCUDA
    process.hltSiPixelClustersSoACUDA = cms.EDProducer("SiPixelRawToClusterCUDAPhase1",
        CablingMapLabel = cms.string(''),
        IncludeErrors = cms.bool(True),
        InputLabel = cms.InputTag("rawDataCollector"),
        Regions = cms.PSet(
            beamSpot = cms.optional.InputTag,
            deltaPhi = cms.optional.vdouble,
            inputs = cms.optional.VInputTag,
            maxZ = cms.optional.vdouble
        ),
        UseQualityInfo = cms.bool(False),
        VCaltoElectronGain = cms.double(1),
        VCaltoElectronGain_L1 = cms.double(1),
        VCaltoElectronOffset = cms.double(0),
        VCaltoElectronOffset_L1 = cms.double(0),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        mightGet = cms.optional.untracked.vstring
    )

    # CUDA EDProducer
    # consumes
    #  - SiPixelClustersCUDA
    # produces
    #  - legacy::SiPixelDigisSoA
    process.hltSiPixelDigisSoACUDA = cms.EDProducer("SiPixelDigisSoAFromCUDA",
        src = cms.InputTag("hltSiPixelClustersSoACUDA")
    )

    # Not sure if digi errors are needed for this comparison
    # CUDA EDProducer
    # consumes
    #  - SiPixelDigiErrorsCUDA
    # produces
    #  - SiPixelErrorsSoA
    process.hltSiPixelDigisErrorsSoACUDA = cms.EDProducer("SiPixelDigiErrorsSoAFromCUDA",
        src = cms.InputTag("hltSiPixelClustersSoACUDA")
    )

    # legacy EDProducer
    # consumes
    #  - legacy::SiPixelDigisSoA
    # produces
    #  - edm::DetSetVector<PixelDigi>
    #  - SiPixelClusterCollectionNew
    process.hltSiPixelClustersCUDA = cms.EDProducer('SiPixelDigisClustersFromSoAPhase1',
        src = cms.InputTag('hltSiPixelDigisSoACUDA'),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        produceDigis = cms.bool(False),
        storeDigis = cms.bool(False)
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
    process.hltSiPixelDigisCUDA = cms.EDProducer('SiPixelDigiErrorsFromSoA',
        digiErrorSoASrc = cms.InputTag('hltSiPixelDigisErrorsSoACUDA'),
        CablingMapLabel = cms.string(''),
        UsePhase1 = cms.bool(True),
        ErrorList = cms.vint32(29),
        UserErrorList = cms.vint32(40)
    )

    # CUDA EDProducer
    # consumes
    #  - BeamSpotCUDA
    #  - SiPixelClustersCUDA
    #  - SiPixelDigisCUDA
    # produces
    #  - TrackingRecHitSoADevice<TrackerTraits>
    process.hltSiPixelRecHitsSoACUDA = cms.EDProducer("SiPixelRecHitCUDAPhase1",
        CPE = cms.string('PixelCPEFast'),
        beamSpot = cms.InputTag("hltOnlineBeamSpotGPU"),
        mightGet = cms.optional.untracked.vstring,
        src = cms.InputTag("hltSiPixelClustersSoACUDA")
    )

    # legacy EDProducer
    # consumes
    #  - TrackingRecHitSoADevice<TrackerTraits>
    #  - SiPixelClusterCollectionNew
    # produces
    #  - SiPixelRecHitCollection
    #  - HMSstorage
    process.hltSiPixelRecHitsCUDA = cms.EDProducer('SiPixelRecHitFromCUDAPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoACUDA'),
        src = cms.InputTag('hltSiPixelClustersCUDA'),
    )

    # Needed to convert hits on device to hits on host for monitoring/comparisons
    # CUDA EDProducer
    # consumes
    #  - TrackingRecHitSoADevice<TrackerTraits>
    # produces
    #  - TrackingRecHitSoAHost<TrackerTraits>
    #  - HMSstorage
    process.hltSiPixelRecHitsSoAFromCUDA = cms.EDProducer('SiPixelRecHitSoAFromCUDAPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoACUDA'),                                             
    )

    ###
    ### Task: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelCUDATask = cms.ConditionalTask(
        process.hltOnlineBeamSpotGPU,
        process.hltSiPixelClustersSoACUDA,
        process.hltSiPixelDigisSoACUDA,
        process.hltSiPixelDigisErrorsSoACUDA,
        process.hltSiPixelClustersCUDA,   # was: hltSiPixelClusters
        process.hltSiPixelDigisCUDA, # was: hltSiPixelDigis
        process.hltSiPixelRecHitsSoACUDA,
        process.hltSiPixelRecHitsSoAFromCUDA,
        process.hltSiPixelRecHitsCUDA,    # was: hltSiPixelRecHits
    )

    process.HLTDoLocalPixelCUDASequence = cms.Sequence( process.HLTDoLocalPixelCUDATask )

    ###
    ### CPUSerial version of Pixel Local Reconstruction
    ###

    # Probably not needed; can keep using hltOnlineBeamSpot
    # CUDA EDProducer
    # consumes
    #  - reco::BeamSpot
    # produces
    #  - BeamSpotCUDA
    process.hltOnlineBeamSpotCPUSerial = process.hltOnlineBeamSpot.clone()

    # legacy EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces
    #  - edm::DetSetVector<PixelDigi>
    #  - edm::DetSetVector<SiPixelRawDataError>
    #  - DetIdCollection
    #  - DetIdCollection
    #  - edmNew::DetSetVector<PixelFEDChannel>
    process.hltSiPixelDigisLegacyFormatCUDACPUSerial = cms.EDProducer("SiPixelRawToDigi",
        IncludeErrors = cms.bool( True ),
        UseQualityInfo = cms.bool( False ),
        ErrorList = cms.vint32( 29 ),
        UserErrorList = cms.vint32(  ),
        InputLabel = cms.InputTag( "rawDataCollector" ),
        Regions = cms.PSet(  ),
        UsePilotBlade = cms.bool( False ),
        UsePhase1 = cms.bool( True ),
        CablingMapLabel = cms.string( "" ),
        SiPixelQualityLabel = cms.string( "" )
    )

    # legacy EDProducer
    # consumes
    #  - edm::DetSetVector<PixelDigi>
    # produces
    #  - SiPixelClusterCollectionNew
    process.hltSiPixelClustersLegacyFormatCUDACPUSerial = cms.EDProducer("SiPixelClusterProducer",
        ChannelThreshold = cms.int32(10),
        ClusterMode = cms.string('PixelThresholdClusterizer'),
        ClusterThreshold = cms.int32(4000),
        ClusterThreshold_L1 = cms.int32(4000),
        DropDuplicates = cms.bool(True),
        ElectronPerADCGain = cms.double(135),
        MissCalibrate = cms.bool(True),
        Phase2Calibration = cms.bool(False),
        Phase2DigiBaseline = cms.double(1200),
        Phase2KinkADC = cms.int32(8),
        Phase2ReadoutMode = cms.int32(-1),
        SeedThreshold = cms.int32(1000),
        SplitClusters = cms.bool(False),
        VCaltoElectronGain = cms.int32(1),
        VCaltoElectronGain_L1 = cms.int32(1),
        VCaltoElectronOffset = cms.int32(0),
        VCaltoElectronOffset_L1 = cms.int32(0),
        maxNumberOfClusters = cms.int32(-1),
        mightGet = cms.optional.untracked.vstring,
        payloadType = cms.string('HLT'),
        src = cms.InputTag("hltSiPixelDigisLegacyFormatCUDACPUSerial")
    )

    # CUDA EDProducer
    # consumes
    #  - reco::BeamSpot
    #  - SiPixelClusterCollectionNew
    # produces
    #  - TrackingRecHitSoAHost<TrackerTraits>
    #  - HMSstorage
    process.hltSiPixelRecHitsSoACUDACPUSerial = cms.EDProducer("SiPixelRecHitSoAFromLegacyPhase1",
        CPE = cms.string('PixelCPEFast'),
        beamSpot = cms.InputTag("hltOnlineBeamSpotCPUSerial"),
        convertToLegacy = cms.bool(False),
        mightGet = cms.optional.untracked.vstring,
        src = cms.InputTag("hltSiPixelClustersLegacyFormatCUDACPUSerial")
    )

    # legacy EDProducer
    # consumes
    #  - TrackingRecHitSoADevice<TrackerTraits>
    #  - SiPixelClusterCollectionNew
    # produces
    #  - SiPixelRecHitCollection
    #  - HMSstorage
    process.hltSiPixelRecHitsLegacyFormatCUDACPUSerial = process.hltSiPixelRecHitsCUDA.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsSoACUDA',
        src = 'hltSiPixelClustersLegacyFormatCUDACPUSerial',
    )

    process.HLTDoLocalPixelCUDACPUSerialTask = cms.ConditionalTask(
        process.hltOnlineBeamSpotCPUSerial,
        process.hltSiPixelDigisLegacyFormatCUDACPUSerial,
        process.hltSiPixelClustersLegacyFormatCUDACPUSerial,
        process.hltSiPixelRecHitsSoACUDACPUSerial,
        process.hltSiPixelRecHitsLegacyFormatCUDACPUSerial,
    )

    process.HLTDoLocalPixelCUDACPUSerialSequence = cms.Sequence( process.HLTDoLocalPixelCUDACPUSerialTask )

    return process

def customizeHLTforCUDAPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in CUDA
    '''

    # CUDA EDProducer
    # consumes
    #  - TrackingRecHitSoADevice<TrackerTraits>
    # produces
    #  - TrackSoAHeterogeneousDevice<TrackerTraits>
    process.hltPixelTracksSoACUDA = cms.EDProducer("CAHitNtupletCUDAPhase1",
        CAThetaCutBarrel = cms.double(0.002),
        CAThetaCutForward = cms.double(0.003),
        dcaCutInnerTriplet = cms.double(0.15),
        dcaCutOuterTriplet = cms.double(0.25),
        doClusterCut = cms.bool(True),
        doPtCut = cms.bool(True),
        doSharedHitCut = cms.bool(True),
        doZ0Cut = cms.bool(True),
        dupPassThrough = cms.bool(False),
        earlyFishbone = cms.bool(True),
        fillStatistics = cms.bool(False),
        fitNas4 = cms.bool(False),
        hardCurvCut = cms.double(0.0328407225),
        idealConditions = cms.bool(False),
        includeJumpingForwardDoublets = cms.bool(True),
        lateFishbone = cms.bool(False),
        maxNumberOfDoublets = cms.uint32(524288),
        minHitsForSharingCut = cms.uint32(10),
        minHitsPerNtuplet = cms.uint32(3),
        onGPU = cms.bool(True),
        phiCuts = cms.vint32(
            522, 730, 730, 522, 626,
            626, 522, 522, 626, 626,
            626, 522, 522, 522, 522,
            522, 522, 522, 522
        ),
        pixelRecHitSrc = cms.InputTag("hltSiPixelRecHitsSoACUDA"),
        ptmin = cms.double(0.9),
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
        useRiemannFit = cms.bool(False),
        useSimpleTripletCleaner = cms.bool(True),
    )

    # CUDA EDProducer
    # consumes
    #  - TrackingRecHitSoAHost<TrackerTraits>
    # produces
    #  - TrackSoAHeterogeneousHost<TrackerTraits>
    process.hltPixelTracksSoACUDACPUSerial = process.hltPixelTracksSoACUDA.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsSoACUDACPUSerial',
        onGPU = cms.bool(False)
    )

    # Needed to convert tracks on device to tracks on host for monitoring/comparisons
    # CUDA EDProducer
    # consumes
    #  - TrackSoAHeterogeneousDevice<TrackerTraits>
    # produces
    #  - TrackSoAHeterogeneousHost<TrackerTraits>
    process.hltPixelTracksSoAFromCUDA = cms.EDProducer( "PixelTrackSoAFromCUDAPhase1",
        src = cms.InputTag( "hltPixelTracksSoACUDA" )
    )

    # legacy EDProducer
    # consumes
    #  - reco::BeamSpot
    #  - TrackSoAHeterogeneousHost<TrackerTraits>
    #  - SiPixelRecHitCollectionNew
    #  - HMSstorage
    # produces
    #  - TrackingRecHitCollection
    #  - reco::TrackExtraCollection
    #  - reco::TrackCollection
    #  - IndToEdm
    process.hltPixelTracksCUDA = cms.EDProducer("PixelTrackProducerFromSoAPhase1",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('loose'),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsCUDA"),
        trackSrc = cms.InputTag("hltPixelTracksSoAFromCUDA")
    )

    # legacy EDProducer
    # consumes
    #  - reco::BeamSpot
    #  - TrackSoAHeterogeneousHost<TrackerTraits>
    #  - SiPixelRecHitCollectionNew
    #  - HMSstorage
    # produces
    #  - TrackingRecHitCollection
    #  - reco::TrackExtraCollection
    #  - reco::TrackCollection
    #  - IndToEdm
    process.hltPixelTracksLegacyFormatCUDACPUSerial = process.hltPixelTracksCUDA.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsLegacyFormatCUDACPUSerial"),
        trackSrc = cms.InputTag("hltPixelTracksSoACUDACPUSerial")
    )

    process.HLTRecoPixelTracksCUDATask = cms.ConditionalTask(
        process.hltPixelTracksSoACUDA,
        process.hltPixelTracksSoAFromCUDA,
        process.hltPixelTracksCUDA,
    )

    process.HLTRecoPixelTracksCUDACPUSerialTask = cms.ConditionalTask(
        process.hltPixelTracksSoACUDACPUSerial,
        process.hltPixelTracksLegacyFormatCUDACPUSerial,
    )

    process.HLTRecoPixelTracksCUDASequence = cms.Sequence( process.HLTRecoPixelTracksCUDATask )

    process.HLTRecoPixelTracksCUDACPUSerialSequence = cms.Sequence( process.HLTRecoPixelTracksCUDACPUSerialTask )

    return process

def customizeHLTforCUDAPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in CUDA
    '''

    # CUDA EDProducer
    # consumes
    #  - TrackSoAHeterogeneousDevice<TrackerTraits>
    # produces
    #  - ZVertexSoADevice
    process.hltPixelVerticesSoACUDA = cms.EDProducer("PixelVertexProducerCUDAPhase1",
        PtMax = cms.double(75),
        PtMin = cms.double(0.5),
        chi2max = cms.double(9),
        eps = cms.double(0.07),
        errmax = cms.double(0.01),
        minT = cms.int32(2),
        onGPU = cms.bool(True),
        oneKernel = cms.bool(True),
        pixelTrackSrc = cms.InputTag("hltPixelTracksSoACUDA"),
        useDBSCAN = cms.bool(False),
        useDensity = cms.bool(True),
        useIterative = cms.bool(False)
    )

    # CUDA EDProducer
    # consumes
    #  - TrackSoAHeterogeneousHost<TrackerTraits>
    # produces
    #  - ZVertexSoAHost
    process.hltPixelVerticesSoACUDACPUSerial = process.hltPixelVerticesSoACUDA.clone(
        onGPU = cms.bool(False),
        pixelTrackSrc = cms.InputTag("hltPixelTracksSoACUDACPUSerial"),
    )

    # Needed to convert vertices on device to vertices on host for monitoring/comparisons
    # CUDA EDProducer
    # consumes
    #  - ZVertexSoADevice
    # produces
    #  - ZVertexSoAHost
    process.hltPixelVerticesSoAFromCUDA = cms.EDProducer( "PixelVertexSoAFromCUDA",
        src = cms.InputTag( "hltPixelVerticesSoACUDA" )
    )

    # legacy EDProducer
    # consumes
    #  - reco::BeamSpot
    #  - ZVertexSoAHost
    #  - reco::TrackCollection
    #  - IndToEdm
    # produces
    #  - reco::VertexCollection
    process.hltPixelVerticesCUDA = cms.EDProducer("PixelVertexProducerFromSoA",
        TrackCollection = cms.InputTag("hltPixelTracksCUDA"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        src = cms.InputTag("hltPixelVerticesSoAFromCUDA")
    )

    # legacy EDProducer
    # consumes
    #  - reco::BeamSpot
    #  - ZVertexSoAHost
    #  - reco::TrackCollection
    #  - IndToEdm
    # produces
    #  - reco::VertexCollection
    process.hltPixelVerticesLegacyFormatCUDACPUSerial = process.hltPixelVerticesCUDA.clone(
        TrackCollection = cms.InputTag("hltPixelTracksLegacyFormatCUDACPUSerial"),
        src = cms.InputTag("hltPixelVerticesSoACUDACPUSerial")
    )

    ## failsafe for fake menus
    if(not hasattr(process,'hltTrimmedPixelVertices')):
        return process

    process.HLTRecopixelvertexingCUDATask = cms.ConditionalTask(
        process.HLTRecoPixelTracksCUDATask,
        process.hltPixelVerticesSoACUDA,
        process.hltPixelVerticesSoAFromCUDA,
        process.hltPixelVerticesCUDA,
        process.hltTrimmedPixelVertices 
    )

    process.HLTRecopixelvertexingCUDACPUSerialTask = cms.ConditionalTask(
        process.HLTRecoPixelTracksCUDACPUSerialTask,
        process.hltPixelVerticesSoACUDACPUSerial,
        process.hltPixelVerticesLegacyFormatCUDACPUSerial,
    )

    process.HLTRecopixelvertexingCUDASequence = cms.Sequence( process.HLTRecopixelvertexingCUDATask )

    process.HLTRecopixelvertexingCUDACPUSerialSequence = cms.Sequence( process.HLTRecopixelvertexingCUDACPUSerialTask )

    return process

# Not sure if this is needed for Alpaka vs CUDA DQM
def customizeHLTforCUDAPixelRecoTheRest(process):
    '''Customize HLT path depending on old SoA tracks
    '''
    process.hltL2TauTagNNProducerCUDA = cms.EDProducer("L2TauNNProducer",
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
        pataTracks = cms.InputTag("hltPixelTracksSoAFromCUDA"),
        pataVertices = cms.InputTag("hltPixelVerticesSoAFromCUDA"),
        track_chi2_max = cms.double(99999.0),
        track_pt_max = cms.double(10.0),
        track_pt_min = cms.double(1.0)
    )
    
    return process

from HLTrigger.Configuration.customizeHLTforAlpaka import customizeHLTforAlpakaPixelRecoLocal
from HLTrigger.Configuration.customizeHLTforAlpaka import customizeHLTforAlpakaPixelRecoTracking
from HLTrigger.Configuration.customizeHLTforAlpaka import customizeHLTforAlpakaPixelRecoVertexing

def customizeHLTforAlpakavsCUDAPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka and CUDA
    '''

    process = customizeHLTforAlpakaPixelRecoLocal(process)
    process = customizeHLTforAlpakaPixelRecoTracking(process)
    process = customizeHLTforAlpakaPixelRecoVertexing(process)
    process = customizeHLTforCUDAPixelRecoLocal(process)
    process = customizeHLTforCUDAPixelRecoTracking(process)
    process = customizeHLTforCUDAPixelRecoVertexing(process)
    process = customizeHLTforDQMAlpakavsCUDAPixel(process)    
    process = customizeHLTforCUDAPixelRecoTheRest(process)

    return process

from HLTrigger.Configuration.customizeHLTforAlpaka import customizeHLTforAlpakaEcalLocalReco
from HLTrigger.Configuration.customizeHLTforAlpaka import customizeHLTforAlpakaParticleFlowClustering

def customizeHLTforAlpakavsCUDA(process):

    process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
    process.load('Configuration.StandardSequences.Accelerators_cff')

    process = customizeHLTforAlpakaEcalLocalReco(process)
    process = customizeHLTforAlpakavsCUDAPixelReco(process)
    process = customizeHLTforAlpakaParticleFlowClustering(process)

    return process

