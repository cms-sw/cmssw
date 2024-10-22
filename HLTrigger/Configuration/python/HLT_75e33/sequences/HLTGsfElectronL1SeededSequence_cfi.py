import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaBestGsfTrackVarsL1Seeded_cfi import *
from ..modules.hltEgammaCkfTrackCandidatesForGSFL1Seeded_cfi import *
from ..modules.hltEgammaGsfElectronsL1Seeded_cfi import *
from ..modules.hltEgammaGsfTracksL1Seeded_cfi import *
from ..modules.hltEgammaGsfTrackVarsL1Seeded_cfi import *

HLTGsfElectronL1SeededSequence = cms.Sequence(hltEgammaCkfTrackCandidatesForGSFL1Seeded+hltEgammaGsfTracksL1Seeded+hltEgammaGsfTrackVarsL1Seeded+hltEgammaGsfElectronsL1Seeded+hltEgammaBestGsfTrackVarsL1Seeded)
