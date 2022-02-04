import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..modules.hltEcalBarrelDigisInRegions_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalPreshowerDigis_cfi import *
from ..modules.hltEcalPreshowerRecHit_cfi import *
from ..modules.hltEcalRecHitL1Seeded_cfi import *
from ..modules.hltEcalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaFilteredCollectionProducer_cfi import *

HLTDoFullUnpackingEgammaEcalL1SeededTask = cms.Task(
    bunchSpacingProducer,
    hltEcalBarrelDigisInRegions,
    hltEcalDetIdToBeRecovered,
    hltEcalDigis,
    hltEcalPreshowerDigis,
    hltEcalPreshowerRecHit,
    hltEcalRecHitL1Seeded,
    hltEcalUncalibRecHitL1Seeded,
    hltL1TEGammaFilteredCollectionProducer
)
