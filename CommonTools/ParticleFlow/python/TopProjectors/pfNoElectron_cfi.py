import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi import tppfCandidatesOnPFCandidates

pfNoElectron = tppfCandidatesOnPFCandidates.clone(
    enable = True,
    name = "noElectron",
    topCollection = "pfIsolatedElectrons",
    bottomCollection = "pfNoMuon",
)

pfNoElectronJME = pfNoElectron.clone(
    bottomCollection = "pfNoMuonJME",
)

pfNoElectronJMEClones = cms.EDProducer("PFCandidateFromFwdPtrProducer",
                                       src=cms.InputTag('pfNoElectronJME')
                                       )
