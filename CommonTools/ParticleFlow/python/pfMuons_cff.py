import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi  import *
from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsFromVertex_cfi import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedMuons_cfi import *


pfMuons = pfIsolatedMuons.clone(cut = "pt > 5 & muonRef.isAvailable()")


pfMuonSequence = cms.Sequence(
    pfAllMuons +
    pfMuonsFromVertex +
    pfIsolatedMuons+
    pfMuons 
    )




