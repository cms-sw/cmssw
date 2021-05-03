import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaBestGsfTrackVarsL1Seeded_cfi import *
from ..modules.hltEgammaCkfTrackCandidatesForGSFL1Seeded_cfi import *
from ..modules.hltEgammaGsfElectronsL1Seeded_cfi import *
from ..modules.hltEgammaGsfTracksL1Seeded_cfi import *
from ..modules.hltEgammaGsfTrackVarsL1Seeded_cfi import *

HLTGsfElectronL1SeededTask = cms.Task(hltEgammaBestGsfTrackVarsL1Seeded, hltEgammaCkfTrackCandidatesForGSFL1Seeded, hltEgammaGsfElectronsL1Seeded, hltEgammaGsfTrackVarsL1Seeded, hltEgammaGsfTracksL1Seeded)
