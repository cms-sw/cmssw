import FWCore.ParameterSet.Config as cms

pfNoElectron = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noElectron"),
    topCollection = cms.InputTag("pfIsolatedElectrons"),
    bottomCollection = cms.InputTag("pfNoMuon"),
)


pfNoElectronJME = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noElectron"),
    topCollection = cms.InputTag("pfIsolatedElectrons"),
    bottomCollection = cms.InputTag("pfNoMuonJME"),
)

pfNoElectronJMEClones = cms.EDProducer("PFCandidateFromFwdPtrProducer",
                                       src=cms.InputTag('pfNoElectronJME')
                                       )
