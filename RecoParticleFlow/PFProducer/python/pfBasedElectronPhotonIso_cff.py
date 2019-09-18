import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoParticleFlow.PFProducer.photonPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.photonPFIsolationValues_cff import *

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


pfBasedElectronPhotonIsoTask = cms.Task(
    pfParticleSelectionTask ,
    pfSelectedElectrons ,
    pfSelectedPhotons ,
    photonPFIsolationDepositsTask ,
    photonPFIsolationValuesTask
    )
pfBasedElectronPhotonIsoSequence = cms.Sequence(pfBasedElectronPhotonIsoTask) 
