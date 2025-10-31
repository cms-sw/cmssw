import FWCore.ParameterSet.Config as cms
import os

def setupLocalInputsForRelVal(local_daq : str = 'local_daq'):

    # Setup DTH source
    inputdir = '/store/group/dpg_hgcal/comm_hgcal/relval'
    files = [
        'run20000000_ls0000_EoR_source2000.jsn',
        'run20000000_ls0000_EoR_source2001.jsn',
        'run20000000_ls0001_EoLS_source2000.jsn',
        'run20000000_ls0001_EoLS_source2001.jsn',
        'run20000000_ls0001_index000000_source2000.raw',
        'run20000000_ls0001_index000000_source2001.raw',
    ]
    localdir = f'{local_daq}/ramdisk/run20000000'
    os.makedirs(localdir, exist_ok=True)
    for f in files:
        os.system(f'xrdcp --silent -f root://cms-xrd-global.cern.ch//{inputdir}/{f} {localdir}')


def runRecoForSep2024TB(process):

    local_daq = setupLocalInputsForRelVal()
    
    process.load('Configuration.StandardSequences.Accelerators_cff')

    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104Reco_cff")
    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104_cff")
    from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
    process = customise_hgcalmapper(
        process, modules='Geometry/HGCalMapping/data/ModuleMaps/modulelocator_P5v1.txt')
    
    process.EvFDaqDirector = cms.Service("EvFDaqDirector",
                                         baseDir=cms.untracked.string('local_daq/fu'),
                                         buBaseDir=cms.untracked.string('local_daq/ramdisk'),
                                         buBaseDirsAll=cms.untracked.vstring('local_daq/ramdisk'),
                                         buBaseDirsNumStreams=cms.untracked.vint32(2),
                                         buBaseDirsStreamIDs=cms.untracked.vint32(2000, 2001),
                                         directorIsBU=cms.untracked.bool(False),
                                         fileBrokerHost=cms.untracked.string('htcp40.cern.ch'),
                                         fileBrokerHostFromCfg=cms.untracked.bool(False),
                                         runNumber=cms.untracked.uint32(20000000),
                                         sourceIdentifier=cms.untracked.string('source'),
                                         useFileBroker=cms.untracked.bool(True)
                                         )

    process.source = cms.Source("DAQSource",
                                dataMode=cms.untracked.string('DTH'),
                                eventChunkBlock=cms.untracked.uint32(100),
                                eventChunkSize=cms.untracked.uint32(100),
                                fileDiscoveryMode=cms.untracked.bool(True),
                                keepRawFiles=cms.untracked.bool(True),
                                maxBufferedFiles=cms.untracked.uint32(2),
                                maxChunkSize=cms.untracked.uint32(1000),
                                numBuffers=cms.untracked.uint32(3),
                                testing=cms.untracked.bool(True),
                                useL1EventID=cms.untracked.bool(False),
                                verifyChecksum=cms.untracked.bool(False)
                                )

    process.FastMonitoringService = cms.Service("FastMonitoringService",
                                                sleepTime=cms.untracked.int32(1)
                                                )

    process.hgcalConfigESProducer = cms.ESSource(
        "HGCalConfigurationESProducer", bePassthroughMode=cms.int32(-1),
        cbHeaderMarker=cms.int32(-1),
        charMode=cms.int32(-1),
        econdHeaderMarker=cms.int32(-1),
        fedjson=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/config_feds_v1.json'),
        indexSource=cms.ESInputTag("hgCalMappingESProducer", ""),
        modjson=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam//config_econds_v1.json'),
        slinkHeaderMarker=cms.int32(-1))

    # Setup HGCal unpacker
    process.hgcalDigis = cms.EDProducer("HGCalRawToDigi",
                                        src=cms.InputTag("rawDataCollector")
                                        )

    # ESProducer to load calibration parameters from JSON file, like pedestals
    process.hgcalCalibParamESProducer = cms.ESProducer(
        'hgcalrechit::HGCalCalibrationESProducer@alpaka',
        filename=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/level0_calib_params_test.json'),
        filenameEnergyLoss=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/hgcal_energyloss_v16.json'),
        indexSource=cms.ESInputTag('hgCalMappingESProducer', ''),
        mapSource=cms.ESInputTag('hgCalMappingModuleESProducer', ''))

    process.hgcalSoARecHits = cms.EDProducer('HGCalRecHitsProducer@alpaka',
                                             digis=cms.InputTag('hgcalDigis', ''),
                                             calibSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
                                             n_hits_scale=cms.int32(1),
                                             n_blocks=cms.int32(1024),
                                             n_threads=cms.int32(1024),
                                             k_noise=cms.double(5.)
                                             )

    from RecoLocalCalo.HGCalRecProducers.hgCalSoARecHitsLayerClustersProducer_cfi import hgCalSoARecHitsLayerClustersProducer
    process.hgcalSoARecHitsLayerClusters = hgCalSoARecHitsLayerClustersProducer.clone(
        hgcalRecHitsSoA="hgcalSoARecHits"
    )

    from RecoLocalCalo.HGCalRecProducers.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
    process.hgcalSoALayerClusters = hgCalSoALayerClustersProducer.clone(
        hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA="hgcalSoARecHits"
    )

    from RecoLocalCalo.HGCalRecProducers.hgCalLayerClustersFromSoAProducer_cfi import hgCalLayerClustersFromSoAProducer
    process.hgcalMergeLayerClusters = hgCalLayerClustersFromSoAProducer.clone(
        hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA="hgcalSoARecHits",
        src="hgcalSoALayerClusters"
    )

    process.reco_task = cms.Task(
        process.hgcalDigis,
        process.hgcalSoARecHits,
        process.hgcalSoARecHitsLayerClusters,
        process.hgcalSoALayerClusters,
        process.hgcalMergeLayerClusters
    )
    process.hgcalTestBeamLocalRecoSequence = cms.Path(process.reco_task)
    process.schedule.insert(0, process.hgcalTestBeamLocalRecoSequence)

    # Keep HGCal output
    process.FEVTDEBUGoutput.outputCommands = ['keep *_hgcal*_*_*']

    return process
