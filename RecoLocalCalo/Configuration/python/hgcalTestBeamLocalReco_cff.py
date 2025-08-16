import FWCore.ParameterSet.Config as cms

def runRecoForSep2024TB(process):
    process.load('Configuration.StandardSequences.Accelerators_cff')

    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104Reco_cff")
    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104_cff")
    from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
    process = customise_hgcalmapper(process, modules='Geometry/HGCalMapping/data/ModuleMaps/modulelocator_Sep2024TBv2.txt')

    # Exclude rawMetaDataCollector from input TB file
    process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_rawMetaDataCollector_*_*')

    # Keep HGCal output
    process.RecoLocalCaloRECO.outputCommands = ['keep *_hgcal*_*_*']

    # ESProducer to load calibration parameters from JSON file, like pedestals
    process.hgcalCalibParamESProducer = cms.ESProducer('hgcalrechit::HGCalCalibrationESProducer@alpaka',
        filename=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/level0_calib_Relay1727033054.json'),
        filenameEnergyLoss=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/hgcal_energyloss_v16.json'),
        indexSource=cms.ESInputTag('hgCalMappingESProducer',''),
        mapSource=cms.ESInputTag('hgCalMappingModuleESProducer','')
    )

    process.hgcalSoARecHits = cms.EDProducer('alpaka_serial_sync::HGCalRecHitsProducer',
        digis=cms.InputTag('hgcalDigis', ''),
        calibSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
        n_hits_scale=cms.int32(1),
        n_blocks=cms.int32(1024),
        n_threads=cms.int32(4096)
    )

    from RecoLocalCalo.HGCalRecProducers.hgCalSoARecHitsLayerClustersProducer_cfi import hgCalSoARecHitsLayerClustersProducer
    process.hgcalSoARecHitsLayerClusters = hgCalSoARecHitsLayerClustersProducer.clone(
        hgcalRecHitsSoA = "hgcalSoARecHits"
    )

    from RecoLocalCalo.HGCalRecProducers.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
    process.hgcalSoALayerClusters = hgCalSoALayerClustersProducer.clone(
        hgcalRecHitsLayerClustersSoA = "hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA = "hgcalSoARecHits"
    )

    from RecoLocalCalo.HGCalRecProducers.hgCalLayerClustersFromSoAProducer_cfi import hgCalLayerClustersFromSoAProducer
    process.hgcalMergeLayerClusters = hgCalLayerClustersFromSoAProducer.clone(
        hgcalRecHitsLayerClustersSoA = "hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA = "hgcalSoARecHits",
        src = "hgcalSoALayerClusters"
    )

    process.reco_task = cms.Task(
        process.hgcalSoARecHits,
        process.hgcalSoARecHitsLayerClusters,
        process.hgcalSoALayerClusters,
        process.hgcalMergeLayerClusters
    )
    process.hgcalTestBeamLocalRecoSequence = cms.Path(process.reco_task)
    process.schedule.insert(0, process.hgcalTestBeamLocalRecoSequence)

    return process
