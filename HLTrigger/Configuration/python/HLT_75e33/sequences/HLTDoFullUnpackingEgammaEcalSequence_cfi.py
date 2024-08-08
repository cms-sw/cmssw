import FWCore.ParameterSet.Config as cms

from ..modules.hltBunchSpacingProducer_cfi import *
from ..modules.hltEcalDetailedTimeRecHit_cfi import *
from ..modules.hltEcalMultiFitUncalibRecHit_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalRecHit_cfi import *

HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(hltEcalDigis+bunchSpacingProducer+hltEcalDetIdToBeRecovered+hltEcalMultiFitUncalibRecHit+hltEcalRecHit+hltEcalDetailedTimeRecHit)
