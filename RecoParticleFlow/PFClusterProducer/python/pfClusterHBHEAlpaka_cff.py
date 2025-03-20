import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.alpaka_cff import alpaka

from RecoParticleFlow.PFRecHitProducer.hcalRecHitSoAProducer_cfi import hcalRecHitSoAProducer as _hcalRecHitSoAProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALParamsESProducer_cfi import pfRecHitHCALParamsESProducer as _pfRecHitHCALParamsESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALTopologyESProducer_cfi import pfRecHitHCALTopologyESProducer as _pfRecHitHCALTopologyESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitSoAProducerHCAL_cfi import pfRecHitSoAProducerHCAL as _pfRecHitSoAProducerHCAL
from RecoParticleFlow.PFRecHitProducer.legacyPFRecHitProducer_cfi import legacyPFRecHitProducer as _legacyPFRecHitProducer
from RecoParticleFlow.PFClusterProducer.pfClusterSoAProducer_cfi import pfClusterSoAProducer as _pfClusterSoAProducer
from RecoParticleFlow.PFClusterProducer.legacyPFClusterProducer_cfi import legacyPFClusterProducer as _legacyPFClusterProducer

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import pfClusteringHBHEHFTask, pfClusteringHBHEHFOnlyTask, particleFlowClusterHBHE, particleFlowRecHitHBHE, particleFlowClusterHCAL, particleFlowClusterHBHEOnly, particleFlowRecHitHBHEOnly, particleFlowClusterHCALOnly
#Full Reco
_alpaka_pfClusteringHBHEHFTask = pfClusteringHBHEHFTask.copy()
#HCAL Only
_alpaka_pfClusteringHBHEHFOnlyTask = pfClusteringHBHEHFOnlyTask.copy()


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

pfRecHitHCALParamsESProducer = _pfRecHitHCALParamsESProducer.clone(
        energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
        energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
    )

pfRecHitHCALTopologyESProducer = _pfRecHitHCALTopologyESProducer.clone()
pfRecHitSoAProducerHCAL = _pfRecHitSoAProducerHCAL.clone(
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hbheRecHitProducerPortable"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:"),
            )
        ),
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.untracked.bool(False)
    )

legacyPFRecHitProducer = _legacyPFRecHitProducer.clone(
        src = "pfRecHitSoAProducerHCAL"
    )

pfClusterSoAProducer = _pfClusterSoAProducer.clone(
        pfRecHits = 'pfRecHitSoAProducerHCAL',
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.bool(False)
    )


legacyPFClusterProducer = _legacyPFClusterProducer.clone(
        src = 'pfClusterSoAProducer',
        pfClusterBuilder = particleFlowClusterHBHE.pfClusterBuilder,
        recHitsSource = 'legacyPFRecHitProducer',
        PFRecHitsLabelIn = 'pfRecHitSoAProducerHCAL'
    )

#Full Reco
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALTopologyRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALTopologyESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitSoAProducerHCAL)
_alpaka_pfClusteringHBHEHFTask.add(legacyPFRecHitProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterSoAProducer)
_alpaka_pfClusteringHBHEHFTask.add(legacyPFClusterProducer)

_alpaka_pfClusteringHBHEHFTask.remove(particleFlowRecHitHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHCAL)
_alpaka_pfClusteringHBHEHFTask.add(particleFlowClusterHCAL)

alpaka.toModify(particleFlowClusterHCAL, clustersSource = "legacyPFClusterProducer")

alpaka.toReplaceWith(pfClusteringHBHEHFTask, _alpaka_pfClusteringHBHEHFTask)

#HCAL Only

pfRecHitSoAProducerHBHEOnly = _pfRecHitSoAProducerHCAL.clone(
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hbheRecHitProducerPortable"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:"),
            )
        ),
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.untracked.bool(False)
    )

legacyPFRecHitProducerHBHEOnly = _legacyPFRecHitProducer.clone(
        src = "pfRecHitSoAProducerHBHEOnly"
    )

pfClusterSoAProducerHBHEOnly = _pfClusterSoAProducer.clone(
        pfRecHits = 'pfRecHitSoAProducerHBHEOnly',
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.bool(False)
    )

legacyPFClusterProducerHBHEOnly = _legacyPFClusterProducer.clone(
        src = 'pfClusterSoAProducerHBHEOnly',
        pfClusterBuilder = particleFlowClusterHBHE.pfClusterBuilder,
        recHitsSource = 'legacyPFRecHitProducerHBHEOnly',
        PFRecHitsLabelIn = 'pfRecHitSoAProducerHBHEOnly'
    )

_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitHCALParamsRecordSource)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitHCALTopologyRecordSource)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitHCALParamsESProducer)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitHCALTopologyESProducer)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitSoAProducerHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.add(legacyPFRecHitProducerHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfClusterSoAProducerHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.add(legacyPFClusterProducerHBHEOnly)

_alpaka_pfClusteringHBHEHFOnlyTask.remove(particleFlowRecHitHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.remove(particleFlowClusterHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.remove(particleFlowClusterHCALOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.add(particleFlowClusterHCALOnly)

alpaka.toModify(particleFlowClusterHCALOnly, clustersSource = "legacyPFClusterProducerHBHEOnly")

alpaka.toReplaceWith(pfClusteringHBHEHFOnlyTask, _alpaka_pfClusteringHBHEHFOnlyTask)
