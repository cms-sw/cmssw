import math
import sys

import FWCore.ParameterSet.Config as cms

# The layer clustering on the CPU (serial_sync backend) of known rechits of a silicon (EE) and a scintillator (BH)
# component, converted to one legacy collection with the barrel ones.
# TestHGCalLayerClustersChecker checks every product against the rechits.
process = cms.Process("TEST")
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
process.options.accelerators = ["cpu"]
process.source = cms.Source("EmptySource")
process.maxEvents.input = 4

def _event(hits):
    """hits: (dim1, dim2, dim3, energy, time, timeError, layer) per rechit, grouped by layer"""
    return cms.PSet(
        dim1 = cms.vdouble([h[0] for h in hits]),
        dim2 = cms.vdouble([h[1] for h in hits]),
        dim3 = cms.vdouble([h[2] for h in hits]),
        energy = cms.vdouble([h[3] for h in hits]),
        sigmaNoise = cms.vdouble([0.001] * len(hits)),
        time = cms.vdouble([h[4] for h in hits]),
        timeError = cms.vdouble([h[5] for h in hits]),
        layer = cms.vint32([h[6] for h in hits]),
        detid = cms.vuint32([1000 + i for i in range(len(hits))]))

_z = 350.
# EE, in (x, y): no rechit; one rechit; two layers with two layer clusters each, the first one with four rechits with
# a time; a layer cluster and an isolated rechit below the seed density on one layer
_eventsEE = [
    _event([]),
    _event([(1., 2., _z, 0.05, 0.1, 0.02, 0)]),
    _event([(0., 0., _z, 0.10, 0.10, 0.02, 0),
            (0.5, 0., _z, 0.05, 0.15, 0.03, 0),
            (10., 0., _z, 0.08, 0.20, -1., 0),
            (0., 0.5, _z, 0.04, 0.12, 0.025, 0),
            (10.6, 0.3, _z, 0.02, 0.30, 0.03, 0),
            (-0.4, 0.2, _z, 0.03, 0.50, 0.02, 0),
            (0., 0., _z + 1., 0.03, 0.1, -1., 1),
            (20., 20., _z + 1., 0.04, 0.1, -1., 1)]),
    _event([(0., 0., _z, 0.05, 0.1, 0.02, 2),
            (50., 50., _z, 0.005, 0.1, 0.02, 2)]),
]
# BH, in (eta, phi) with phi in [0, 2 pi): no rechit; one rechit; two rechits on the two sides of phi = 0 and one
# far away; one rechit on each of two layers
_eventsBH = [
    _event([]),
    _event([(2., 1., _z, 0.05, 0., -1., 30)]),
    _event([(2., 0.005, _z, 0.05, 0., -1., 30),
            (2., 2. * math.pi - 0.005, _z, 0.04, 0., -1., 30),
            (2., 3., _z, 0.03, 0., -1., 30)]),
    _event([(1.8, 1., _z, 0.05, 0., -1., 30),
            (1.8, 1., _z + 5., 0.05, 0., -1., 31)]),
]

process.recHitsEE = cms.EDProducer("TestHGCalSoARecHitsSource", events = cms.VPSet(*_eventsEE))
process.recHitsBH = cms.EDProducer("TestHGCalSoARecHitsSource", events = cms.VPSet(*_eventsBH))
# with the argument badLayerSizes, the layer sizes of the EE rechits count one rechit too many, and the job must fail
if "badLayerSizes" in sys.argv:
    process.recHitsEE.layerSizesExcess = cms.uint32(1)

from RecoTICL.LayerClustering.hgCalCLUEsteringLayerClustersProducer_cfi import hgCalCLUEsteringLayerClustersProducer
from RecoTICL.LayerClustering.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
from RecoTICL.LayerClustering.hgCalMergedLayerClustersFromSoAProducer_cfi import hgCalMergedLayerClustersFromSoAProducer

process.clusteringEE = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = "recHitsEE", detector = "EE", deltac = 1.3, kappa = 9., outlierDeltaFactor = 2.,
    extremesFromGeometry = False)
process.clusteringBH = hgCalCLUEsteringLayerClustersProducer.clone(
    hgcalRecHitsSoA = "recHitsBH", detector = "BH", deltac = 0.0315, kappa = 9., outlierDeltaFactor = 2.,
    extremesFromGeometry = False)
process.layerClustersEE = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = "recHitsEE", hgcalRecHitsLayerClustersSoA = "clusteringEE", detector = "EE")
process.layerClustersBH = hgCalSoALayerClustersProducer.clone(
    hgcalRecHitsSoA = "recHitsBH", hgcalRecHitsLayerClustersSoA = "clusteringBH", detector = "BH")
process.barrelLayerClusters = cms.EDProducer("TestLegacyLayerClustersSource")
process.mergedLayerClusters = hgCalMergedLayerClustersFromSoAProducer.clone(
    layerClustersSoA = ["layerClustersEE", "layerClustersBH"],
    hgcalRecHitsSoA = ["recHitsEE", "recHitsBH"],
    hgcalRecHitsLayerClustersSoA = ["clusteringEE", "clusteringBH"],
    extraLayerClusters = ["barrelLayerClusters"],
    extraTimeLayerClusters = ["barrelLayerClusters:timeLayerCluster"])

process.check = cms.EDAnalyzer("TestHGCalLayerClustersChecker",
    components = cms.VPSet(
        cms.PSet(recHits = cms.InputTag("recHitsEE"), clustering = cms.InputTag("clusteringEE"),
                 layerClusters = cms.InputTag("layerClustersEE"), algoId = cms.int32(6),
                 isScintillator = cms.bool(False), expectedNumberOfClusters = cms.vint32(0, 1, 4, 1)),
        cms.PSet(recHits = cms.InputTag("recHitsBH"), clustering = cms.InputTag("clusteringBH"),
                 layerClusters = cms.InputTag("layerClustersBH"), algoId = cms.int32(8),
                 isScintillator = cms.bool(True), expectedNumberOfClusters = cms.vint32(0, 1, 2, 2))),
    mergedLayerClusters = cms.InputTag("mergedLayerClusters"),
    extraLayerClusters = cms.InputTag("barrelLayerClusters"))

process.task = cms.Task(process.recHitsEE, process.recHitsBH, process.clusteringEE, process.clusteringBH,
                        process.layerClustersEE, process.layerClustersBH,
                        process.barrelLayerClusters, process.mergedLayerClusters)
process.path = cms.Path(process.check, process.task)
