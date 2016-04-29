import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

hgcalLocalRecoSequence = cms.Sequence(HGCalUncalibRecHit+HGCalRecHit)
