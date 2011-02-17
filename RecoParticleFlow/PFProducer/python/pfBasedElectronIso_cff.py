import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfCandsForIsolation_cff  import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *

pfSelectedElectrons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==11")
)

pfBasedElectronIsoSequence = cms.Sequence(
    pfCandsForIsolationSequence +
    pfSelectedElectrons +
    pfElectronIsolationSequence
    ) 
