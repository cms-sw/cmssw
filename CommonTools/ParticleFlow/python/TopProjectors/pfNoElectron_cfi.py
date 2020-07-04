import FWCore.ParameterSet.Config as cms
import CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi as _mod

pfNoElectron = _mod.tppfCandidatesOnPFCandidates.clone(
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
