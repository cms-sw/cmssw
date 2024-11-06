import FWCore.ParameterSet.Config as cms

from ..modules.hltBunchSpacingProducer_cfi import *
from ..modules.hltEcalBarrelDigisInRegions_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalRecHitL1Seeded_cfi import *
from ..modules.hltEcalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaFilteredCollectionProducer_cfi import *

HLTDoFullUnpackingEgammaEcalL1SeededSequence = cms.Sequence(hltEcalDigis+bunchSpacingProducer+hltEcalDetIdToBeRecovered+hltL1TEGammaFilteredCollectionProducer+hltEcalBarrelDigisInRegions+hltEcalUncalibRecHitL1Seeded+hltEcalRecHitL1Seeded)
