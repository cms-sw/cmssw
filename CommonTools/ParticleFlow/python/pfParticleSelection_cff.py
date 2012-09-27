import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import *

pfParticleSelectionSequence = cms.Sequence(
    pfNoPileUpIsoSequence +
    pfSortByTypeSequence 
    )
