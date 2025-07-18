import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalMultiFitUncalibRecHit_cfi import *
from ..modules.hltEcalRecHit_cfi import *
from ..modules.hltFixedGridRhoFastjetAllCaloForEGamma_cfi import *

HLTFastJetForEgammaSequence = cms.Sequence(hltEcalMultiFitUncalibRecHit+hltEcalRecHit+hltFixedGridRhoFastjetAllCaloForEGamma)
