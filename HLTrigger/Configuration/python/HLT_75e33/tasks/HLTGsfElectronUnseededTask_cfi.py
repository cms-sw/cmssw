import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaBestGsfTrackVarsUnseeded_cfi import *
from ..modules.hltEgammaCkfTrackCandidatesForGSFUnseeded_cfi import *
from ..modules.hltEgammaGsfElectronsUnseeded_cfi import *
from ..modules.hltEgammaGsfTracksUnseeded_cfi import *
from ..modules.hltEgammaGsfTrackVarsUnseeded_cfi import *

HLTGsfElectronUnseededTask = cms.Task(hltEgammaBestGsfTrackVarsUnseeded, hltEgammaCkfTrackCandidatesForGSFUnseeded, hltEgammaGsfElectronsUnseeded, hltEgammaGsfTrackVarsUnseeded, hltEgammaGsfTracksUnseeded)
