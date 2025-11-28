import FWCore.ParameterSet.Config as cms


#from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *
#from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cff import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowBadHcalPseudoCluster_cff import *

particleFlowClusterECALTask = cms.Task(particleFlowClusterECAL)
particleFlowClusterECALSequence = cms.Sequence(particleFlowClusterECALTask)

pfClusteringECALTask = cms.Task(particleFlowRecHitECAL,
                                particleFlowClusterECALUncorrected,
                                particleFlowClusterECALTask)
pfClusteringECAL = cms.Sequence(pfClusteringECALTask)

pfClusteringPSTask = cms.Task(particleFlowRecHitPS,particleFlowClusterPS)
pfClusteringPS = cms.Sequence(pfClusteringPSTask)

pfClusteringHBHEHFTask = cms.Task(particleFlowRecHitHBHE,
                                  particleFlowRecHitHF,
                                  particleFlowClusterHBHE,
                                  particleFlowClusterHF,
                                  particleFlowClusterHCAL)
pfClusteringHBHEHF = cms.Sequence(pfClusteringHBHEHFTask)

pfClusteringHBHEHFOnlyTask = cms.Task(particleFlowRecHitHBHEOnly,
                                      particleFlowRecHitHF,
                                      particleFlowClusterHBHEOnly,
                                      particleFlowClusterHF,
                                      particleFlowClusterHCALOnly)

#--- Legacy HCAL Only Task
# as more non-legacy modules are added, make legacy copies and add them here
pfClusteringHBHEHFOnlyLegacyTask = cms.Task(particleFlowRecHitHBHEOnlyLegacy,
                                            particleFlowRecHitHF,
                                            particleFlowClusterHBHEOnlyLegacy,
                                            particleFlowClusterHF,
                                            particleFlowClusterHCALOnly)

pfClusteringHOTask = cms.Task(particleFlowRecHitHO,particleFlowClusterHO)
pfClusteringHO = cms.Sequence(pfClusteringHOTask)

particleFlowClusterWithoutHOTask = cms.Task(particleFlowBadHcalPseudoCluster,
                                            pfClusteringPSTask,
                                            pfClusteringECALTask,
                                            pfClusteringHBHEHFTask)
particleFlowClusterWithoutHO = cms.Sequence(particleFlowClusterWithoutHOTask)

particleFlowClusterTask = cms.Task(particleFlowBadHcalPseudoCluster,
                                   pfClusteringPSTask,
                                   pfClusteringECALTask,
                                   pfClusteringHBHEHFTask,
                                   pfClusteringHOTask)
particleFlowCluster = cms.Sequence(particleFlowClusterTask)

#HGCal

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cfi import particleFlowRecHitHGC
pfClusteringHGCalTask = cms.Task(particleFlowRecHitHGC)
pfClusteringHGCal = cms.Sequence(pfClusteringHGCalTask)

_phase2_hgcal_particleFlowClusterTask = particleFlowClusterTask.copy()
_phase2_hgcal_particleFlowClusterTask.add(pfClusteringHGCalTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( particleFlowClusterTask, _phase2_hgcal_particleFlowClusterTask )

#timing

from RecoParticleFlow.PFClusterProducer.particleFlowClusterTimeAssigner_cfi import particleFlowTimeAssignerECAL
from RecoParticleFlow.PFSimProducer.ecalBarrelClusterFastTimer_cfi import ecalBarrelClusterFastTimer
_phase2_timing_particleFlowClusterECALTask = particleFlowClusterECALTask.copy()
_phase2_timing_particleFlowClusterECALTask.add(cms.Task(ecalBarrelClusterFastTimer,
                                                        particleFlowTimeAssignerECAL))

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith(particleFlowClusterECALTask,
                                  _phase2_timing_particleFlowClusterECALTask)
phase2_timing.toModify(particleFlowClusterECAL,
                            inputECAL = 'particleFlowTimeAssignerECAL')

# Replace HBHE rechit and clustering with Alpaka modules

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from RecoParticleFlow.PFRecHitProducer.hcalRecHitSoAProducer_cfi import hcalRecHitSoAProducer as _hcalRecHitSoAProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALParamsESProducer_cfi import pfRecHitHCALParamsESProducer as _pfRecHitHCALParamsESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALTopologyESProducer_cfi import pfRecHitHCALTopologyESProducer as _pfRecHitHCALTopologyESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitSoAProducerHCAL_cfi import pfRecHitSoAProducerHCAL as _pfRecHitSoAProducerHCAL
from RecoParticleFlow.PFClusterProducer.pfClusterSoAProducer_cfi import pfClusterSoAProducer as _pfClusterSoAProducer

_alpaka_pfClusteringHBHEHFTask = pfClusteringHBHEHFTask.copy() #Full Reco
_alpaka_pfClusteringHBHEHFOnlyTask = pfClusteringHBHEHFOnlyTask.copy() #HCAL Only


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
        appendToDataLabel = cms.string("offline"),
        energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
        energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
    )

pfRecHitHCALTopologyESProducer = _pfRecHitHCALTopologyESProducer.clone(
        appendToDataLabel = cms.string("offline")
    )


alpaka_pfClusteringHBHEHF_esProducersTask = cms.Task(pfRecHitHCALParamsRecordSource,
                                                     pfRecHitHCALTopologyRecordSource,
                                                     pfRecHitHCALParamsESProducer,
                                                     pfRecHitHCALTopologyESProducer
                                            )





pfRecHitSoAProducerHCAL = _pfRecHitSoAProducerHCAL.clone(
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hbheRecHitProducerPortable"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:offline"),
            )
        ),
        topology = "pfRecHitHCALTopologyESProducer:offline",
        synchronise = cms.untracked.bool(False)
    )


pfClusterSoAProducer = _pfClusterSoAProducer.clone(
        pfRecHits = 'pfRecHitSoAProducerHCAL',
        topology = "pfRecHitHCALTopologyESProducer:offline",
        synchronise = cms.bool(False)
    )

_alpaka_pfClusteringHBHEHFTask.add(alpaka_pfClusteringHBHEHF_esProducersTask)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitSoAProducerHCAL)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterSoAProducer)

pfRecHitSoAProducerHBHEOnly = _pfRecHitSoAProducerHCAL.clone(
        producers = [
            cms.PSet(
                src = cms.InputTag("hbheRecHitProducerPortable"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:offline"),
            )
        ],
        topology = "pfRecHitHCALTopologyESProducer:offline",
    )

pfClusterSoAProducerHBHEOnly = _pfClusterSoAProducer.clone(
        pfRecHits = 'pfRecHitSoAProducerHBHEOnly',
        topology = "pfRecHitHCALTopologyESProducer:offline",
    )

_alpaka_pfClusteringHBHEHFOnlyTask.add(alpaka_pfClusteringHBHEHF_esProducersTask)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfRecHitSoAProducerHBHEOnly)
_alpaka_pfClusteringHBHEHFOnlyTask.add(pfClusterSoAProducerHBHEOnly)

alpaka.toReplaceWith(pfClusteringHBHEHFTask, _alpaka_pfClusteringHBHEHFTask)
alpaka.toReplaceWith(pfClusteringHBHEHFOnlyTask, _alpaka_pfClusteringHBHEHFOnlyTask)

from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cff import barrelLayerClustersEB, barrelLayerClustersHB
_pfClusteringECALTask = pfClusteringECALTask.copy()
_pfClusteringECALTask.add(barrelLayerClustersEB)

_pfClusteringHBHEHFTask = pfClusteringHBHEHFTask.copy()
_pfClusteringHBHEHFTask.add(barrelLayerClustersHB)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(pfClusteringECALTask, _pfClusteringECALTask)
ticl_barrel.toReplaceWith(pfClusteringHBHEHFTask, _pfClusteringHBHEHFTask)

