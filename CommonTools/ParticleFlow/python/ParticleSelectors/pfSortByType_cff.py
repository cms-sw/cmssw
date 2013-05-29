import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadrons_cfi  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedHadrons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllPhotons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi import *

from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedParticles_cfi import *

from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadronsAndPhotons_cfi import *

pfPileUpAllChargedParticles = pfAllChargedParticles.clone( src = 'pfPileUpIso' )


pfSortByTypeSequence = cms.Sequence(
    pfAllNeutralHadrons+
    pfAllChargedHadrons+
    pfAllPhotons+
    # charged hadrons + electrons + muons
    pfAllChargedParticles+
    # same, but from pile up
    pfPileUpAllChargedParticles+
    pfAllNeutralHadronsAndPhotons
#    +
#    pfAllElectrons+
#    pfAllMuons
    )

