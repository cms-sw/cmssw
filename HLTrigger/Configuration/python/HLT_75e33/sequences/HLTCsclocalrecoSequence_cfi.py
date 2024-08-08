import FWCore.ParameterSet.Config as cms

from ..modules.hltCsc2DRecHits_cfi import *
from ..modules.hltCscSegments_cfi import *

HLTCsclocalrecoSequence = cms.Sequence(hltCsc2DRecHits+hltCscSegments)
