import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUp_cff  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *

from CommonTools.ParticleFlow.pfPileUpCandidates_cff import *


pfCandsForIsolationSequence = cms.Sequence(
    pfNoPileUpSequence + 
    pfSortByTypeSequence +
    pfPileUpCandidatesSequence
    )
