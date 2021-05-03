import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalPreshowerDigis_cfi import *
from ..modules.hltEcalPreshowerRecHit_cfi import *
from ..modules.hltEcalRecHit_cfi import *
from ..modules.hltEcalUncalibRecHit_cfi import *

HLTDoFullUnpackingEgammaEcalTask = cms.Task(bunchSpacingProducer, hltEcalDetIdToBeRecovered, hltEcalDigis, hltEcalPreshowerDigis, hltEcalPreshowerRecHit, hltEcalRecHit, hltEcalUncalibRecHit)
