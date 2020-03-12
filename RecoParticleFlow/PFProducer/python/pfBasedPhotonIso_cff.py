import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoParticleFlow.PFProducer.photonPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.photonPFIsolationValues_cff import *

pfSelectedPhotons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("pdgId()==22 && mva_nothing_gamma>0")
#    cut = cms.string("pdgId()==22")
)

pfBasedPhotonIsoTask = cms.Task(
    pfParticleSelectionTask,
    pfSelectedPhotons,
    photonPFIsolationDepositsTask,
    photonPFIsolationValuesTask 
) 
pfBasedPhotonIsoSequence = cms.Sequence(pfBasedPhotonIsoTask)
