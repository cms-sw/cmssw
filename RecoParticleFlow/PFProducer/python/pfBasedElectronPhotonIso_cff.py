import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolationFromDeposits_cff import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolationFromDeposits_cff import *

pfSelectedElectrons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlowTmp"),
    cut = cms.string("abs(pdgId())==11")
)

pfSelectedPhotons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlowTmp"),
    cut = cms.string("pdgId()==22 && mva_nothing_gamma>0")
)


# pfPileUp.PFCandidates = cms.InputTag("particleFlowTmp")
# pfNoPileUp.bottomCollection = cms.InputTag("particleFlowTmp") 

pfBasedElectronPhotonIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    pfSelectedElectrons +
    pfElectronIsolationSequence+
    pfElectronIsolationFromDepositsSequence+
    pfSelectedPhotons +
    pfPhotonIsolationSequence+
    pfPhotonIsolationFromDepositsSequence
    ) 
