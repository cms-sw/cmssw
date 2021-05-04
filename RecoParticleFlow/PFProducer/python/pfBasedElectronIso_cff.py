import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

pfSelectedElectrons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==11")
)

pfBasedElectronIsoTask = cms.Task(
    pfParticleSelectionTask,
    pfSelectedElectrons,
    electronPFIsolationDepositsTask,
    electronPFIsolationValuesTask
    )
pfBasedElectronIsoSequence = cms.Sequence(pfBasedElectronIsoTask)

#COLIN: is this file used in RECO? in PF2PAT? same for photons. 
