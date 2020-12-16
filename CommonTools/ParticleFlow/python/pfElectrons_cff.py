import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *


pfElectrons = pfIsolatedElectrons.clone( cut = " pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.hitPattern().numberOfLostHits('MISSING_INNER_HITS')<2")

pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    pfElectronsFromVertex + 
    pfIsolatedElectrons +    
    pfElectrons 
    )




