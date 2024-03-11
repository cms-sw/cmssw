import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..modules.ecalDetailedTimeRecHit_cfi import *
from ..modules.ecalMultiFitUncalibRecHit_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalRecHit_cfi import *
from ..modules.hltEcalUncalibRecHit_cfi import *

HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(hltEcalDigis+bunchSpacingProducer+hltEcalDetIdToBeRecovered+hltEcalUncalibRecHit+ecalMultiFitUncalibRecHit+hltEcalRecHit+ecalDetailedTimeRecHit)
