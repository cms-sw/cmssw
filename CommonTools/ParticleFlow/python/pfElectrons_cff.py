import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedElectrons_cfi import *


pfElectrons = pfIsolatedElectrons.clone( cut = cms.string(" pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits<2"))

pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    pfElectronsFromVertex + 
    pfIsolatedElectrons +    
    pfElectrons 
    )




