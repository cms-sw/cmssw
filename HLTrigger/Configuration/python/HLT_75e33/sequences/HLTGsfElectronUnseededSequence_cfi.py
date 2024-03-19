import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaBestGsfTrackVarsUnseeded_cfi import *
from ..modules.hltEgammaCkfTrackCandidatesForGSFUnseeded_cfi import *
from ..modules.hltEgammaGsfElectronsUnseeded_cfi import *
from ..modules.hltEgammaGsfTracksUnseeded_cfi import *
from ..modules.hltEgammaGsfTrackVarsUnseeded_cfi import *

HLTGsfElectronUnseededSequence = cms.Sequence(hltEgammaCkfTrackCandidatesForGSFUnseeded+hltEgammaGsfTracksUnseeded+hltEgammaGsfElectronsUnseeded+hltEgammaBestGsfTrackVarsUnseeded+hltEgammaGsfTrackVarsUnseeded)
