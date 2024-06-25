import FWCore.ParameterSet.Config as cms
import re
import itertools

from FWCore.ParameterSet.MassReplace import massReplaceInputTag
from HeterogeneousCore.AlpakaCore.functions import *
from HLTrigger.Configuration.common import *

## PF HLT in Alpaka
def customizeHLTforAlpakaParticleFlowClustering(process):
    '''Customization to introduce Particle Flow Reconstruction in Alpaka
    '''
    ## failsafe for fake menus
    if not hasattr(process, 'hltParticleFlowClusterHBHE'):
        return process

    for prod in producers_by_type(process, 'HCALRecHitSoAProducer@alpaka'):
        return process

    process.hltESSPFRecHitHCALParamsRecord = cms.ESSource('EmptyESSource',
        recordName = cms.string('PFRecHitHCALParamsRecord'),
        iovIsRunNotTime = cms.bool(True),
        firstValid = cms.vuint32(1)
    )

    process.hltESSPFRecHitHCALTopologyRecord = cms.ESSource('EmptyESSource',
        recordName = cms.string('PFRecHitHCALTopologyRecord'),
        iovIsRunNotTime = cms.bool(True),
        firstValid = cms.vuint32(1)
    )

    process.hltESSJobConfigurationGPURecord = cms.ESSource('EmptyESSource',
        recordName = cms.string('JobConfigurationGPURecord'),
        iovIsRunNotTime = cms.bool(True),
        firstValid = cms.vuint32(1)
    )

    process.hltHbheRecHitSoA = cms.EDProducer("HCALRecHitSoAProducer@alpaka",
        src = cms.InputTag("hltHbhereco"),
        synchronise = cms.untracked.bool(False),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPPFRecHitHCALTopology = cms.ESProducer('PFRecHitHCALTopologyESProducer@alpaka',
        usePFThresholdsFromDB = cms.bool(True),
        appendToDataLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPPFRecHitHCALParams = cms.ESProducer('PFRecHitHCALParamsESProducer@alpaka',
        energyThresholdsHB = cms.vdouble(0.1, 0.2, 0.3, 0.3),
        energyThresholdsHE = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        appendToDataLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltParticleFlowRecHitHBHESoA = cms.EDProducer("PFRecHitSoAProducerHCAL@alpaka",
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hltHbheRecHitSoA"),
                params = cms.ESInputTag("hltESPPFRecHitHCALParams:"),
            )
        ),
        topology = cms.ESInputTag("hltESPPFRecHitHCALTopology:"),
        synchronise = cms.untracked.bool(False),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltParticleFlowRecHitHBHE = cms.EDProducer("LegacyPFRecHitProducer",
        src = cms.InputTag("hltParticleFlowRecHitHBHESoA")
    )

    process.hltESPPFClusterParams = cms.ESProducer("PFClusterParamsESProducer@alpaka",
        seedFinder = cms.PSet(
            nNeighbours = cms.int32(4),
            thresholdsByDetector = cms.VPSet(
                cms.PSet(
                    detector = cms.string('HCAL_BARREL1'),
                    seedingThreshold = cms.vdouble(0.125, 0.25, 0.35, 0.35),
                    seedingThresholdPt = cms.double(0)
                ),
                cms.PSet(
                    detector = cms.string('HCAL_ENDCAP'),
                    seedingThreshold = cms.vdouble(0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275),
                    seedingThresholdPt = cms.double(0)
                )
            )
        ),
        initialClusteringStep = cms.PSet(
            thresholdsByDetector = cms.VPSet(
                cms.PSet(
                    detector = cms.string('HCAL_BARREL1'),
                    gatheringThreshold = cms.vdouble(0.1, 0.2, 0.3, 0.3)
                ),
                cms.PSet(
                    detector = cms.string('HCAL_ENDCAP'),
                    gatheringThreshold = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
                )
            )
        ),
        pfClusterBuilder = cms.PSet(
            maxIterations = cms.uint32(5),
            minFracTot = cms.double(1e-20),
            minFractionToKeep = cms.double(1e-07),
            excludeOtherSeeds = cms.bool(True),
            showerSigma = cms.double(10),
            stoppingTolerance = cms.double(1e-08),
            recHitEnergyNorms = cms.VPSet(
                cms.PSet(
                    detector = cms.string('HCAL_BARREL1'),
                    recHitEnergyNorm = cms.vdouble(0.1, 0.2, 0.3, 0.3)
                ),
                cms.PSet(
                    detector = cms.string('HCAL_ENDCAP'),
                    recHitEnergyNorm = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
                )
            ),
            positionCalc = cms.PSet(
                minFractionInCalc = cms.double(1e-09),
                minAllowedNormalization = cms.double(1e-09)
            ),
            timeResolutionCalcBarrel = cms.PSet(
                corrTermLowE = cms.double(0),
                threshLowE = cms.double(6),
                noiseTerm = cms.double(21.86),
                constantTermLowE = cms.double(4.24),
                noiseTermLowE = cms.double(8),
                threshHighE = cms.double(15),
                constantTerm = cms.double(2.82)
            ),
            timeResolutionCalcEndcap = cms.PSet(
                corrTermLowE = cms.double(0),
                threshLowE = cms.double(6),
                noiseTerm = cms.double(21.86),
                constantTermLowE = cms.double(4.24),
                noiseTermLowE = cms.double(8),
                threshHighE = cms.double(15),
                constantTerm = cms.double(2.82)
            )
        ),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltParticleFlowClusterHBHESoA = cms.EDProducer("PFClusterSoAProducer@alpaka",
        pfRecHits = cms.InputTag("hltParticleFlowRecHitHBHESoA"),
        topology = cms.ESInputTag("hltESPPFRecHitHCALTopology:"),
        pfClusterParams = cms.ESInputTag("hltESPPFClusterParams:"),
        synchronise = cms.bool(False),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltParticleFlowClusterHBHE = cms.EDProducer("LegacyPFClusterProducer",
        src = cms.InputTag("hltParticleFlowClusterHBHESoA"),
        pfClusterBuilder = process.hltParticleFlowClusterHBHE.pfClusterBuilder,
        usePFThresholdsFromDB = cms.bool(True),
        recHitsSource = cms.InputTag("hltParticleFlowRecHitHBHE"),
        PFRecHitsLabelIn = cms.InputTag("hltParticleFlowRecHitHBHESoA")
    )

    process.HLTPFHcalClustering = cms.Sequence(
        process.hltHbheRecHitSoA +
        process.hltParticleFlowRecHitHBHESoA +
        process.hltParticleFlowRecHitHBHE +
        process.hltParticleFlowClusterHBHESoA +
        process.hltParticleFlowClusterHBHE +
        process.hltParticleFlowClusterHCAL
    )

    # Some Sequences contain all the modules of process.HLTPFHcalClustering Sequence instead of the Sequence itself
    # find these Sequences and replace all the modules with the Sequence
    def replaceItemsInSequence(process, seqNames, itemsToReplace, replacingSequence):
        for seqName in seqNames:
            if not hasattr(process, seqName):
                continue
            seq = getattr(process, seqName)
            for item in itemsToReplace:
                # remove items that will be replaced by replacingSequence
                if (item != itemsToReplace[-1]):
                    seq.remove(item)
                else:
                    # if last item, replace it with the Sequence
                    seq.replace(item, replacingSequence)
        return process

    process = replaceItemsInSequence(
        process, [
            'HLTParticleFlowSequence',
            'HLTParticleFlowSequenceForTaus',
            'HLTParticleFlowSequenceForDisplTaus',
            'HLTParticleFlowSequencePPOnAA',
            'HLTPixelOnlyParticleFlowSequence',
        ], [
            process.hltParticleFlowRecHitHBHE,
            process.hltParticleFlowClusterHBHE,
            process.hltParticleFlowClusterHCAL
        ],
        process.HLTPFHcalClustering
    )

    process.hltHbheRecHitSoACPUSerial = makeSerialClone(process.hltHbheRecHitSoA)

    process.hltParticleFlowRecHitHBHESoACPUSerial = makeSerialClone(process.hltParticleFlowRecHitHBHESoA)
    process.hltParticleFlowRecHitHBHESoACPUSerial.producers[0].src = 'hltHbheRecHitSoACPUSerial'

    process.hltParticleFlowRecHitHBHECPUOnly = process.hltParticleFlowRecHitHBHE.clone(
        src = 'hltParticleFlowRecHitHBHESoACPUSerial',
    )

    process.hltParticleFlowClusterHBHESoACPUSerial = makeSerialClone(process.hltParticleFlowClusterHBHESoA,
        pfRecHits = 'hltParticleFlowRecHitHBHESoACPUSerial',
    )

    process.hltParticleFlowClusterHBHECPUOnly = process.hltParticleFlowClusterHBHE.clone(
        src = 'hltParticleFlowClusterHBHESoACPUSerial',
        recHitsSource = 'hltParticleFlowRecHitHBHECPUOnly',
        PFRecHitsLabelIn = 'hltParticleFlowRecHitHBHESoACPUSerial',
    )

    process.HLTPFHcalClusteringCPUOnly = cms.Sequence(
        process.hltHbheRecHitSoACPUSerial +
        process.hltParticleFlowRecHitHBHESoACPUSerial +
        process.hltParticleFlowRecHitHBHECPUOnly +
        process.hltParticleFlowClusterHBHESoACPUSerial +
        process.hltParticleFlowClusterHBHECPUOnly +
        process.hltParticleFlowClusterHCALCPUOnly
    )

    process = replaceItemsInSequence(process, ['HLTParticleFlowCPUOnlySequence'],
        [process.hltParticleFlowRecHitHBHECPUOnly, process.hltParticleFlowClusterHBHECPUOnly, process.hltParticleFlowClusterHCALCPUOnly],
        process.HLTPFHcalClusteringCPUOnly)

    # modify EventContent of *DQMGPUvsCPU streams
    for hltOutModMatch in ['hltOutputDQMGPUvsCPU', 'hltOutputHIDQMGPUvsCPU']:
        if hasattr(process, hltOutModMatch):
            outMod = getattr(process, hltOutModMatch)
            outMod.outputCommands.extend([
                'keep *_hltParticleFlowClusterHBHESoA_*_*',
                'keep *_hltParticleFlowClusterHBHESoACPUSerial_*_*',
            ])

    # Add PF sequences to DQM_*HcalReconstruction_v Path
    for pathNameMatch in ['DQM_HcalReconstruction_v', 'DQM_HIHcalReconstruction_v']:
        dqmHcalRecoPathName = None
        for pathName in process.paths_():
            if pathName.startswith(pathNameMatch):
                dqmHcalRecoPathName = pathName
                break
        if dqmHcalRecoPathName == None:
            continue
        dqmHcalPath = getattr(process, dqmHcalRecoPathName)
        dqmHcalRecoPathIndex = dqmHcalPath.index(process.hltHcalConsumerGPU) + 1
        dqmHcalPath.insert(dqmHcalRecoPathIndex, process.HLTPFHcalClusteringCPUOnly)
        dqmHcalPath.insert(dqmHcalRecoPathIndex, process.HLTPFHcalClustering)

    return process


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

    for prod in producers_by_type(process, 'SiPixelPhase1MonitorRecHitsSoAAlpaka'):
        return process

    # modify EventContent of DQMGPUvsCPU stream
    try:
        outCmds_new = [foo for foo in process.hltOutputDQMGPUvsCPU.outputCommands if 'Pixel' not in foo]
        outCmds_new += [
            'keep *Cluster*_hltSiPixelClusters_*_*',
            'keep *Cluster*_hltSiPixelClustersSerialSync_*_*',
            'keep *_hltSiPixelDigiErrors_*_*',
            'keep *_hltSiPixelDigiErrorsSerialSync_*_*',
            'keep *RecHit*_hltSiPixelRecHits_*_*',
            'keep *RecHit*_hltSiPixelRecHitsSerialSync_*_*',
            'keep *_hltPixelTracks_*_*',
            'keep *_hltPixelTracksSerialSync_*_*',
            'keep *_hltPixelVertices_*_*',
            'keep *_hltPixelVerticesSerialSync_*_*',
        ]
        process.hltOutputDQMGPUvsCPU.outputCommands = outCmds_new[:]
    except:
        pass

    # PixelRecHits: monitor of SerialSync product (Alpaka backend: 'serial_sync')
    process.hltPixelRecHitsSoAMonitorCPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag('hltSiPixelRecHitsSoASerialSync'),
        TopFolderName = cms.string('SiPixelHeterogeneous/PixelRecHitsCPU')
    )

    # PixelRecHits: monitor of GPU product (Alpaka backend: '')
    process.hltPixelRecHitsSoAMonitorGPU = cms.EDProducer('SiPixelPhase1MonitorRecHitsSoAAlpaka',
        pixelHitsSrc = cms.InputTag('hltSiPixelRecHitsSoA'),
        TopFolderName = cms.string('SiPixelHeterogeneous/PixelRecHitsGPU')
    )

    # PixelRecHits: 'GPUvsCPU' comparisons
    process.hltPixelRecHitsSoACompareGPUvsCPU = cms.EDProducer('SiPixelPhase1CompareRecHitsSoAAlpaka',
        pixelHitsSrcHost = cms.InputTag('hltSiPixelRecHitsSoASerialSync'),
        pixelHitsSrcDevice = cms.InputTag('hltSiPixelRecHitsSoA'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU'),
        minD2cut = cms.double(1.0e-4)
    )

    process.hltPixelTracksSoAMonitorCPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoASerialSync'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelTracksSoAMonitorGPU = cms.EDProducer("SiPixelPhase1MonitorTrackSoAAlpaka",
        minQuality = cms.string('loose'),
        pixelTrackSrc = cms.InputTag('hltPixelTracksSoA'),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackGPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelTracksSoACompareGPUvsCPU = cms.EDProducer("SiPixelPhase1CompareTrackSoAAlpaka",
        deltaR2cut = cms.double(0.04),
        minQuality = cms.string('loose'),
        pixelTrackSrcHost = cms.InputTag("hltPixelTracksSoASerialSync"),
        pixelTrackSrcDevice = cms.InputTag("hltPixelTracksSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU'),
        useQualityCut = cms.bool(True)
    )

    process.hltPixelVertexSoAMonitorCPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoASerialSync"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexCPU')
    )

    process.hltPixelVertexSoAMonitorGPU = cms.EDProducer("SiPixelMonitorVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        pixelVertexSrc = cms.InputTag("hltPixelVerticesSoA"),
        topFolderName = cms.string('SiPixelHeterogeneous/PixelVertexGPU')
    )

    process.hltPixelVertexSoACompareGPUvsCPU = cms.EDProducer("SiPixelCompareVertexSoAAlpaka",
        beamSpotSrc = cms.InputTag("hltOnlineBeamSpot"),
        dzCut = cms.double(1),
        pixelVertexSrcHost = cms.InputTag("hltPixelVerticesSoASerialSync"),
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

    for delMod in ['hltPixelConsumerCPU', 'hltPixelConsumerGPU']:
        if hasattr(process, delMod):
            process.__delattr__(delMod)

    return process


def customizeHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTDoLocalPixelSequence'):
        return process

    for prod in producers_by_type(process, 'SiPixelRawToClusterPhase1@alpaka'):
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

    if hasattr(process, 'hltSiPixelDigisSoA'):
        del process.hltSiPixelDigisSoA
    if hasattr(process, 'hltSiPixelDigiErrorsSoA'):
        del process.hltSiPixelDigiErrorsSoA

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
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltSiPixelRecHitsGPU'):
        del process.hltSiPixelRecHitsGPU
    if hasattr(process, 'hltSiPixelRecHitsFromGPU'):
        del process.hltSiPixelRecHitsFromGPU
    if hasattr(process, 'hltSiPixelRecHitsSoAFromGPU'):
        del process.hltSiPixelRecHitsSoAFromGPU

    process.hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase1',
        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHitsSoA'),
        src = cms.InputTag('hltSiPixelClusters'),
    )

    ###
    ### Sequence: Pixel Local Reconstruction
    ###
    process.HLTDoLocalPixelSequence = cms.Sequence(
        process.hltOnlineBeamSpotDevice +
        process.hltSiPixelClustersSoA +
        process.hltSiPixelClusters +
        process.hltSiPixelDigiErrors +    # renamed from hltSiPixelDigis
        process.hltSiPixelRecHitsSoA +
        process.hltSiPixelRecHits
    )

    # The module hltSiPixelClustersCache is only used in the PRef menu.
    # Previously, it being included in the GRun menu had no effect
    # because it was included in a ConditionalTask, which meant that
    # it was not executed in the GRun menu, since no other modules
    # in any Path of that menu were consuming it.
    # Since this customisation effectively replaces ConditionalTasks in the HLT menus with Sequences,
    # leaving hltSiPixelClustersCache in a Sequence of the GRun menu would lead to it being unnecessarily executed.
    # On the other hand, hltSiPixelClustersCache must be kept in the PRef menu,
    # as the latter does have reconstruction modules which consume it.
    # The solution implemented here is
    # (a) not to include hltSiPixelClustersCache in the Sequence HLTDoLocalPixelSequence used in the GRun menu, and
    # (b) include hltSiPixelClustersCache in a separate Sequence to be used only as part of the PRef menu.
    if hasattr(process, 'HLTPixelClusterSplittingForPFPPRefForDmeson') \
        and not hasattr(process, 'HLTDoLocalPixelSequenceForPFPPRefForDmeson') \
        and process.HLTPixelClusterSplittingForPFPPRefForDmeson.contains(process.HLTDoLocalPixelSequence):

        process.HLTDoLocalPixelSequenceForPFPPRefForDmeson = cms.Sequence(
            process.HLTDoLocalPixelSequence +
            process.hltSiPixelClustersCache
        )

        process.HLTPixelClusterSplittingForPFPPRefForDmeson.insert(
            process.HLTPixelClusterSplittingForPFPPRefForDmeson.index(process.HLTDoLocalPixelSequence),
            process.HLTDoLocalPixelSequenceForPFPPRefForDmeson
        )

    # remove HLTDoLocalPixelTask (not needed anymore)
    if hasattr(process, 'HLTDoLocalPixelTask'):
        del process.HLTDoLocalPixelTask

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

    process.hltSiPixelRecHitsSoASerialSync = makeSerialClone(process.hltSiPixelRecHitsSoA,
        beamSpot = 'hltOnlineBeamSpotDeviceSerialSync',
        src = 'hltSiPixelClustersSoASerialSync',
    )

    process.hltSiPixelRecHitsSerialSync = process.hltSiPixelRecHits.clone(
        pixelRecHitSrc = 'hltSiPixelRecHitsSoASerialSync',
        src = 'hltSiPixelClustersSerialSync',
    )

    process.HLTDoLocalPixelCPUOnlySequence = cms.Sequence(
        process.hltOnlineBeamSpotDeviceSerialSync +
        process.hltSiPixelClustersSoASerialSync +
        process.hltSiPixelClustersSerialSync +
        process.hltSiPixelDigiErrorsSerialSync +
        process.hltSiPixelRecHitsSoASerialSync +
        process.hltSiPixelRecHitsSerialSync
    )

    if hasattr(process, 'HLTDoLocalPixelCPUOnlyTask'):
        del process.HLTDoLocalPixelCPUOnlyTask

    if hasattr(process, 'hltMeasurementTrackerEventCPUOnly'):
        process.hltMeasurementTrackerEventCPUOnly.pixelClusterProducer = "hltSiPixelClustersSerialSync"
        process.hltMeasurementTrackerEventCPUOnly.inactivePixelDetectorLabels = ["hltSiPixelDigiErrorsSerialSync"]
        process.hltMeasurementTrackerEventCPUOnly.badPixelFEDChannelCollectionLabels = ["hltSiPixelDigiErrorsSerialSync"]

    if hasattr(process, 'hltDoubletRecoveryClustersRefRemovalCPUOnly'):
        process.hltDoubletRecoveryClustersRefRemovalCPUOnly.pixelClusters = "hltSiPixelClustersSerialSync"

    if hasattr(process, 'hltDoubletRecoveryPFlowPixelClusterCheckCPUOnly'):
        process.hltDoubletRecoveryPFlowPixelClusterCheckCPUOnly.PixelClusterCollectionLabel = "hltSiPixelClustersSerialSync"

    if hasattr(process, 'hltDoubletRecoveryPixelLayersAndRegionsCPUOnly'):
        process.hltDoubletRecoveryPixelLayersAndRegionsCPUOnly.inactivePixelDetectorLabels = ['hltSiPixelDigiErrorsSerialSync']
        process.hltDoubletRecoveryPixelLayersAndRegionsCPUOnly.badPixelFEDChannelCollectionLabels = ['hltSiPixelDigiErrorsSerialSync']
        process.hltDoubletRecoveryPixelLayersAndRegionsCPUOnly.BPix.HitProducer = "hltSiPixelRecHitsSerialSync"
        process.hltDoubletRecoveryPixelLayersAndRegionsCPUOnly.FPix.HitProducer = "hltSiPixelRecHitsSerialSync"

    if hasattr(process, 'hltIter3IterL3FromL1MuonClustersRefRemovalCPUOnly'):
        process.hltIter3IterL3FromL1MuonClustersRefRemovalCPUOnly.pixelClusters = "hltSiPixelClustersSerialSync"

    if hasattr(process, 'hltIter3IterL3FromL1MuonPixelClusterCheckCPUOnly'):
        process.hltIter3IterL3FromL1MuonPixelClusterCheckCPUOnly.PixelClusterCollectionLabel = "hltSiPixelClustersSerialSync"

    if hasattr(process, 'hltIter3IterL3FromL1MuonPixelLayersAndRegionsCPUOnly'):
        process.hltIter3IterL3FromL1MuonPixelLayersAndRegionsCPUOnly.inactivePixelDetectorLabels = ['hltSiPixelDigiErrorsSerialSync']
        process.hltIter3IterL3FromL1MuonPixelLayersAndRegionsCPUOnly.badPixelFEDChannelCollectionLabels = ['hltSiPixelDigiErrorsSerialSync']
        process.hltIter3IterL3FromL1MuonPixelLayersAndRegionsCPUOnly.BPix.HitProducer = "hltSiPixelRecHitsSerialSync"
        process.hltIter3IterL3FromL1MuonPixelLayersAndRegionsCPUOnly.FPix.HitProducer = "hltSiPixelRecHitsSerialSync"

    for modLabel in [
        'hltDoubletRecoveryPixelLayersAndRegions',
        'hltFullIter6PixelTrackingRegionSeedLayersBPPRef',
        'hltIter3IterL3FromL1MuonPixelLayersAndRegions',
        'hltMeasurementTrackerEvent',
    ]:
        if hasattr(process, modLabel):
            mod = getattr(process, modLabel)
            mod.inactivePixelDetectorLabels = ['hltSiPixelDigiErrors']
            mod.badPixelFEDChannelCollectionLabels = ['hltSiPixelDigiErrors']

    return process


def customizeHLTforAlpakaPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTRecoPixelTracksSequence'):
        return process

    for prod in producers_by_type(process, 'CAHitNtupletAlpakaPhase1@alpaka'):
        return process

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

    if hasattr(process, 'hltL2TauTagNNProducer'):
        process.hltL2TauTagNNProducer = cms.EDProducer("L2TauNNProducerAlpaka", **process.hltL2TauTagNNProducer.parameters_())

    process.hltPixelTracksSoASerialSync = makeSerialClone(process.hltPixelTracksSoA,
        pixelRecHitSrc = 'hltSiPixelRecHitsSoASerialSync'
    )

    process.hltPixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase1",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        minQuality = cms.string('loose'),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
        trackSrc = cms.InputTag("hltPixelTracksSoA")
    )

    if hasattr(process, 'hltPixelTracksCPU'):
        del process.hltPixelTracksCPU
    if hasattr(process, 'hltPixelTracksCPUOnly'):
        del process.hltPixelTracksCPUOnly
    if hasattr(process, 'hltPixelTracksFromGPU'):
        del process.hltPixelTracksFromGPU
    if hasattr(process, 'hltPixelTracksGPU'):
        del process.hltPixelTracksGPU

    process.hltPixelTracksSerialSync = process.hltPixelTracks.clone(
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHitsSerialSync"),
        trackSrc = cms.InputTag("hltPixelTracksSoASerialSync")
    )

    process.HLTRecoPixelTracksSequence = cms.Sequence(
        process.hltPixelTracksSoA +
        process.hltPixelTracks
    )

    if hasattr(process, 'HLTRecoPixelTracksTask'):
        del process.HLTRecoPixelTracksTask

    process.HLTRecoPixelTracksSerialSyncSequence = cms.Sequence(
        process.hltPixelTracksSoASerialSync +
        process.hltPixelTracksSerialSync
    )

    if hasattr(process, 'HLTRecoPixelTracksCPUOnlyTask'):
        del process.HLTRecoPixelTracksCPUOnlyTask

    process.hltPixelTracksInRegionL2CPUOnly.tracks = "hltPixelTracksSerialSync"

    process.hltPixelTracksInRegionL1CPUOnly.tracks = "hltPixelTracksSerialSync"

    process.hltIter0PFLowPixelSeedsFromPixelTracksCPUOnly.InputCollection = "hltPixelTracksSerialSync"

    return process


def customizeHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''

    if not hasattr(process, 'HLTRecopixelvertexingSequence'):
        return process

    # do not apply the customisation if the menu is already using the alpaka pixel reconstruction
    for prod in producers_by_type(process, 'PixelVertexProducerAlpakaPhase1@alpaka'):
        return process

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
    if not hasattr(process, 'hltTrimmedPixelVertices'):
        return process

    process.HLTRecopixelvertexingSequence = cms.Sequence(
        process.HLTRecoPixelTracksSequence +
        process.hltPixelVerticesSoA +
        process.hltPixelVertices +
        process.hltTrimmedPixelVertices
    )

    if hasattr(process, 'HLTRecopixelvertexingTask'):
        del process.HLTRecopixelvertexingTask

    process.HLTRecopixelvertexingCPUOnlySequence = cms.Sequence(
        process.HLTRecoPixelTracksSerialSyncSequence +
        process.hltPixelVerticesSoASerialSync +
        process.hltPixelVerticesSerialSync +
        process.hltTrimmedPixelVerticesCPUOnly
    )

    if hasattr(process, 'HLTRecopixelvertexingCPUOnlyTask'):
        del process.HLTRecopixelvertexingCPUOnlyTask

    process.hltTrimmedPixelVerticesCPUOnly.src = 'hltPixelVerticesSerialSync'
    process.hltParticleFlowCPUOnly.vertexCollection = 'hltPixelVerticesSerialSync'
    process.hltAK4PFJetsCPUOnly.srcPVs = 'hltPixelVerticesSerialSync'

    return process


def customizeHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''
    process = customizeHLTforAlpakaPixelRecoLocal(process)
    process = customizeHLTforAlpakaPixelRecoTracking(process)
    process = customizeHLTforAlpakaPixelRecoVertexing(process)
    process = customizeHLTforDQMGPUvsCPUPixel(process)

    return process


## ECAL HLT in Alpaka
def customizeHLTforAlpakaEcalLocalReco(process):

    if not hasattr(process, 'hltEcalDigisGPU'):
        return process

    for prod in producers_by_type(process, 'EcalRawToDigiPortable@alpaka'):
        return process

    # remove existing ECAL GPU-related ES modules
    for foo in [foo for foo in process.es_producers_() if ('ecal' in foo and 'GPU' in foo)]:
        process.__delattr__(foo)

    for foo in [foo for foo in process.es_sources_() if ('ecal' in foo and 'GPU' in foo)]:
        process.__delattr__(foo)

    # redefine ECAL local reconstruction sequence
    process.hltEcalDigisPortableSoA = cms.EDProducer("EcalRawToDigiPortable@alpaka",
        FEDs = process.hltEcalDigisGPU.FEDs,
        InputLabel = process.hltEcalDigisGPU.InputLabel,
        digisLabelEB = process.hltEcalDigisGPU.digisLabelEB,
        digisLabelEE = process.hltEcalDigisGPU.digisLabelEE,
        maxChannelsEB = process.hltEcalDigisGPU.maxChannelsEB,
        maxChannelsEE = process.hltEcalDigisGPU.maxChannelsEE,
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    from EventFilter.EcalRawToDigi.ecalElectronicsMappingHostESProducer_cfi import ecalElectronicsMappingHostESProducer as _ecalElectronicsMappingHostESProducer
    process.ecalElectronicsMappingHostESProducer = _ecalElectronicsMappingHostESProducer.clone()

    process.hltEcalDigis = cms.EDProducer("EcalDigisFromPortableProducer",
        digisInLabelEB = cms.InputTag('hltEcalDigisPortableSoA', 'ebDigis'),
        digisInLabelEE = cms.InputTag('hltEcalDigisPortableSoA', 'eeDigis'),
        digisOutLabelEB = cms.string("ebDigis"),
        digisOutLabelEE = cms.string("eeDigis"),
        produceDummyIntegrityCollections = cms.bool(False)
    )

    process.hltEcalUncalibRecHitPortableSoA = cms.EDProducer("EcalUncalibRecHitProducerPortable@alpaka",
        EBtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EBtimeConstantTerm,
        EBtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Lower,
        EBtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Upper,
        EBtimeNconst = process.hltEcalUncalibRecHitGPU.EBtimeNconst,
        EEtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EEtimeConstantTerm,
        EEtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Lower,
        EEtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Upper,
        EEtimeNconst = process.hltEcalUncalibRecHitGPU.EEtimeNconst,
        amplitudeThresholdEB = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEB,
        amplitudeThresholdEE = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEE,
        digisLabelEB = cms.InputTag("hltEcalDigisPortableSoA", "ebDigis"),
        digisLabelEE = cms.InputTag("hltEcalDigisPortableSoA", "eeDigis"),
        kernelMinimizeThreads = process.hltEcalUncalibRecHitGPU.kernelMinimizeThreads,
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
        shouldRunTimingComputation = process.hltEcalUncalibRecHitGPU.shouldRunTimingComputation,
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    if hasattr(process, 'hltEcalUncalibRecHitGPU'):
        del process.hltEcalUncalibRecHitGPU

    process.ecalMultifitParametersSource = cms.ESSource("EmptyESSource",
        firstValid = cms.vuint32(1),
        iovIsRunNotTime = cms.bool(True),
        recordName = cms.string('EcalMultifitParametersRcd')
    )

    from RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi import ecalMultifitConditionsHostESProducer as _ecalMultifitConditionsHostESProducer
    process.ecalMultifitConditionsHostESProducer = _ecalMultifitConditionsHostESProducer.clone()

    from RecoLocalCalo.EcalRecProducers.ecalMultifitParametersHostESProducer_cfi import ecalMultifitParametersHostESProducer as _ecalMultifitParametersHostESProducer
    process.ecalMultifitParametersHostESProducer = _ecalMultifitParametersHostESProducer.clone()

    process.hltEcalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitSoAToLegacy",
        isPhase2 = process.hltEcalUncalibRecHitFromSoA.isPhase2,
        recHitsLabelCPUEB = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEB,
        recHitsLabelCPUEE = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEE,
        uncalibRecHitsPortableEB = cms.InputTag("hltEcalUncalibRecHitPortableSoA", "EcalUncalibRecHitsEB"),
        uncalibRecHitsPortableEE = cms.InputTag("hltEcalUncalibRecHitPortableSoA", "EcalUncalibRecHitsEE")
    )

    if hasattr(process, 'hltEcalUncalibRecHitSoA'):
        delattr(process, 'hltEcalUncalibRecHitSoA')

    process.hltEcalDetIdToBeRecovered = cms.EDProducer("EcalDetIdToBeRecoveredProducer",
        integrityBlockSizeErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityBlockSizeErrors'),
        integrityTTIdErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityTTIdErrors'),

        ebIntegrityGainErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityGainErrors'),
        eeIntegrityGainErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityGainErrors'),

        ebIntegrityGainSwitchErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityGainSwitchErrors'),
        eeIntegrityGainSwitchErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityGainSwitchErrors'),

        ebIntegrityChIdErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityChIdErrors'),
        eeIntegrityChIdErrors = cms.InputTag('hltEcalDigisLegacy', 'EcalIntegrityChIdErrors'),

        ebSrFlagCollection = cms.InputTag("hltEcalDigisLegacy"),
        eeSrFlagCollection = cms.InputTag("hltEcalDigisLegacy"),

        ebDetIdToBeRecovered = cms.string("ebDetId"),
        eeDetIdToBeRecovered = cms.string("eeDetId"),

        ebFEToBeRecovered = cms.string("ebFE"),
        eeFEToBeRecovered = cms.string("eeFE"),
    )

    process.hltEcalRecHit.triggerPrimitiveDigiCollection = 'hltEcalDigisLegacy:EcalTriggerPrimitives'

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence(
        process.hltEcalDigisLegacy +
        process.hltEcalDigisPortableSoA +
        process.hltEcalDigis +          # conversion of PortableSoA to legacy format
        process.hltEcalUncalibRecHitPortableSoA +
        process.hltEcalUncalibRecHit +  # conversion of PortableSoA to legacy format
        process.hltEcalDetIdToBeRecovered +
        process.hltEcalRecHit
    )

    process.HLTPreshowerSequence = cms.Sequence(process.hltEcalPreshowerDigis + process.hltEcalPreshowerRecHit)

    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence +
        process.HLTPreshowerSequence
    )

    process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence(process.HLTDoFullUnpackingEgammaEcalSequence)

    process.hltEcalDigisCPUSerialSoA = makeSerialClone(process.hltEcalDigisPortableSoA)

    process.hltEcalDigisCPUSerial = process.hltEcalDigis.clone(
        digisInLabelEB = 'hltEcalDigisCPUSerialSoA:ebDigis',
        digisInLabelEE = 'hltEcalDigisCPUSerialSoA:eeDigis',
    )

    process.hltEcalUncalibRecHitCPUSerialSoA = makeSerialClone(process.hltEcalUncalibRecHitPortableSoA,
        digisLabelEB = "hltEcalDigisCPUSerialSoA:ebDigis",
        digisLabelEE = "hltEcalDigisCPUSerialSoA:eeDigis",
    )

    process.hltEcalUncalibRecHitCPUSerial = process.hltEcalUncalibRecHit.clone(
        uncalibRecHitsPortableEB = "hltEcalUncalibRecHitCPUSerialSoA:EcalUncalibRecHitsEB",
        uncalibRecHitsPortableEE = "hltEcalUncalibRecHitCPUSerialSoA:EcalUncalibRecHitsEE",
    )

    process.hltEcalRecHitCPUOnly = process.hltEcalRecHit.clone(
        EBuncalibRecHitCollection = 'hltEcalUncalibRecHitCPUSerial:EcalUncalibRecHitsEB',
        EEuncalibRecHitCollection = 'hltEcalUncalibRecHitCPUSerial:EcalUncalibRecHitsEE',
    )

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerCPUOnlySequence = cms.Sequence(
        process.hltEcalDigisLegacy +
        process.hltEcalDigisCPUSerialSoA +
        process.hltEcalDigisCPUSerial + # conversion of CPUSerialSoA to legacy format
        process.hltEcalUncalibRecHitCPUSerialSoA +
        process.hltEcalUncalibRecHitCPUSerial + # conversion of CPUSerialSoA to legacy format
        process.hltEcalDetIdToBeRecovered +
        process.hltEcalRecHitCPUOnly
    )

    for prod in producers_by_type(process, 'HLTRechitsToDigis'):
        prod.srFlagsIn = 'hltEcalDigisLegacy'

    for prod in producers_by_type(process, 'CorrectedECALPFClusterProducer'):
        try:
            prod.energyCorrector.ebSrFlagLabel = 'hltEcalDigisLegacy'
            prod.energyCorrector.eeSrFlagLabel = 'hltEcalDigisLegacy'
        except:
            pass

    for pathNameMatch in ['DQM_EcalReconstruction_v', 'DQM_HIEcalReconstruction_v']:
        dqmEcalRecoPathName = None
        for pathName in process.paths_():
            if pathName.startswith(pathNameMatch):
                dqmEcalRecoPath = getattr(process, pathName)
                dqmEcalRecoPath.insert(dqmEcalRecoPath.index(process.HLTEndSequence), getattr(process, 'HLTDoFullUnpackingEgammaEcalWithoutPreshowerCPUOnlySequence'))
                for delmod in ['hltEcalConsumerCPU', 'hltEcalConsumerGPU']:
                    if hasattr(process, delmod):
                        process.__delattr__(delmod)

    for hltOutModMatch in ['hltOutputDQMGPUvsCPU', 'hltOutputHIDQMGPUvsCPU']:
        if hasattr(process, hltOutModMatch):
            outMod = getattr(process, hltOutModMatch)
            outCmds_new = [foo for foo in outMod.outputCommands if 'Ecal' not in foo]
            outCmds_new += [
                'keep *_hltEcalDigis_*_*',
                'keep *_hltEcalDigisCPUSerial_*_*',
                'keep *_hltEcalUncalibRecHit_*_*',
                'keep *_hltEcalUncalibRecHitCPUSerial_*_*',
            ]
            outMod.outputCommands = outCmds_new[:]

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


def customizeHLTforAlpaka(process):
    process.load('Configuration.StandardSequences.Accelerators_cff')

    process = customizeHLTforAlpakaStatus(process)
    process = customizeHLTforAlpakaPixelReco(process)
    process = customizeHLTforAlpakaEcalLocalReco(process)
    process = customizeHLTforAlpakaParticleFlowClustering(process)
    process = customizeHLTforAlpakaRename(process)

    return process
