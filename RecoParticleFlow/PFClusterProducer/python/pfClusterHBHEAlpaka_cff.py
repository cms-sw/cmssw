import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.alpaka_cff import alpaka

from RecoParticleFlow.PFRecHitProducer.hcalRecHitSoAProducer_cfi import hcalRecHitSoAProducer as _hcalRecHitSoAProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALParamsESProducer_cfi import pfRecHitHCALParamsESProducer as _pfRecHitHCALParamsESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALTopologyESProducer_cfi import pfRecHitHCALTopologyESProducer as _pfRecHitHCALTopologyESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitSoAProducerHCAL_cfi import pfRecHitSoAProducerHCAL as _pfRecHitSoAProducerHCAL
from RecoParticleFlow.PFRecHitProducer.legacyPFRecHitProducer_cfi import legacyPFRecHitProducer as _legacyPFRecHitProducer
from RecoParticleFlow.PFClusterProducer.pfClusterParamsESProducer_cfi import pfClusterParamsESProducer as _pfClusterParamsESProducer
from RecoParticleFlow.PFClusterProducer.pfClusterSoAProducer_cfi import pfClusterSoAProducer as _pfClusterSoAProducer
from RecoParticleFlow.PFClusterProducer.legacyPFClusterProducer_cfi import legacyPFClusterProducer as _legacyPFClusterProducer

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import pfClusteringHBHEHFTask, particleFlowClusterHBHE, particleFlowRecHitHBHE, particleFlowClusterHCAL

_alpaka_pfClusteringHBHEHFTask = pfClusteringHBHEHFTask.copy()

pfRecHitHCALParamsRecordSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('PFRecHitHCALParamsRecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

pfRecHitHCALTopologyRecordSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('PFRecHitHCALTopologyRecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

pfClusterParamsRecordSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('JobConfigurationGPURecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

hbheRecHitToSoA = _hcalRecHitSoAProducer.clone(
        src = "hbhereco"
    )

pfRecHitHCALParamsESProducer = _pfRecHitHCALParamsESProducer.clone(
        energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
        energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
    )

pfRecHitHCALTopologyESProducer = _pfRecHitHCALTopologyESProducer.clone()
pfRecHitSoAProducerHCAL = _pfRecHitSoAProducerHCAL.clone(
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hbheRecHitToSoA"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:"),
            )
        ),
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.untracked.bool(False)
    )

legacyPFRecHitProducer = _legacyPFRecHitProducer.clone(
        src = "pfRecHitSoAProducerHCAL"
    )

pfClusterParamsESProducer = _pfClusterParamsESProducer.clone()
pfClusterSoAProducer = _pfClusterSoAProducer.clone(
        pfRecHits = 'pfRecHitSoAProducerHCAL',
        topology = "pfRecHitHCALTopologyESProducer:",
        pfClusterParams = 'pfClusterParamsESProducer:',
        synchronise = cms.bool(False)
    )


legacyPFClusterProducer = _legacyPFClusterProducer.clone(
        src = 'pfClusterSoAProducer',
        pfClusterParams = 'pfClusterParamsESProducer:',
        pfClusterBuilder = particleFlowClusterHBHE.pfClusterBuilder,
        recHitsSource = 'legacyPFRecHitProducer',
        PFRecHitsLabelIn = 'pfRecHitSoAProducerHCAL'
    )


_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALTopologyRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterParamsRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(hbheRecHitToSoA)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitSoAProducerHCAL)
_alpaka_pfClusteringHBHEHFTask.add(legacyPFRecHitProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterParamsESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterSoAProducer)

_alpaka_pfClusteringHBHEHFTask.remove(particleFlowRecHitHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHCAL)
_alpaka_pfClusteringHBHEHFTask.add(particleFlowClusterHBHE)
_alpaka_pfClusteringHBHEHFTask.add(particleFlowClusterHCAL)

alpaka.toReplaceWith(particleFlowClusterHBHE, legacyPFClusterProducer)

alpaka.toReplaceWith(pfClusteringHBHEHFTask, _alpaka_pfClusteringHBHEHFTask)
