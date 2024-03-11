import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCkfTrackCandidatesForGSFL1Seeded_cfi import *
from ..modules.hltEgammaGsfTracksL1Seeded_cfi import *
from ..modules.hltEgammaGsfTrackVarsL1Seeded_cfi import *

HLTEle115NonIsoL1SeededGsfElectronL1SeededSequence = cms.Sequence(hltEgammaCkfTrackCandidatesForGSFL1Seeded+hltEgammaGsfTracksL1Seeded+hltEgammaGsfTrackVarsL1Seeded)
