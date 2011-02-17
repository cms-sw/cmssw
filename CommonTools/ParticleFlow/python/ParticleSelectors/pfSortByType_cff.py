import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadrons_cfi  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedHadrons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllPhotons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi import *

pfSortByTypeSequence = cms.Sequence(
    pfAllNeutralHadrons+
    pfAllChargedHadrons+
    pfAllPhotons
#    +
#    pfAllElectrons+
#    pfAllMuons
    )

