import FWCore.ParameterSet.Config as cms

#HGCAL reconstruction
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cff import hgcalMergeLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import *

HGCalLocalRecoTestBeamTask = cms.Task(HGCalUncalibRecHit,HGCalRecHit,hgcalMergeLayerClusters,hgcalMultiClusters)
HGCalLocalRecoTestBeamSequence = cms.Sequence(HGCalLocalRecoTestBeamTask)
