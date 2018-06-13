import FWCore.ParameterSet.Config as cms

#HGCAL reconstruction
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import *

HGCalLocalRecoTestBeamSequence = cms.Sequence(HGCalUncalibRecHit*HGCalRecHit*hgcalLayerClusters*hgcalMultiClusters)
