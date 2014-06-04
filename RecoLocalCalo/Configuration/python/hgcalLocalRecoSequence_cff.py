import FWCore.ParameterSet.Config as cms

#HGCAL local reconstruction
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

hgcalUncalibRecHitSequence = cms.Sequence(HGCalUncalibRecHit
                                      )

hgcalRecHitSequence        = cms.Sequence(HGCalRecHit
                                         )

hgcalLocalRecoSequence     = cms.Sequence(hgcalUncalibRecHitSequence*
                                         hgcalRecHitSequence)

