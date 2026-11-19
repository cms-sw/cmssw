import FWCore.ParameterSet.Config as cms

from ..modules.hltMkFitSiPixelHits_cfi import *
from ..modules.hltMkFitSiPhase2Hits_cfi import *
from ..modules.hltMkFitEventOfHits_cfi import *

HLTMkFitInputSequence = cms.Sequence(
    hltMkFitSiPixelHits
    +hltMkFitSiPhase2Hits
    +hltMkFitEventOfHits
)
