import FWCore.ParameterSet.Config as cms

l1TauRecoTree = cms.EDAnalyzer("L1TauRecoTreeProducer",
   period                          = cms.string("2015"),
   maxTau                          = cms.uint32(20),
   PFTauTag                        = cms.untracked.InputTag("hpsPFTauProducer"),
   PFTauDMFindingOld               = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs"),
   PFTauDMFinding                  = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
   PFTauTightIsoTag                = cms.untracked.InputTag("hpsPFTauDiscriminationByTightIsolation"),
   PFTauLooseIsoTag                = cms.untracked.InputTag("hpsPFTauDiscriminationByLooseIsolation"),
   PFTauLooseAntiMuon              = cms.untracked.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"),
   PFTauTightAntiMuon              = cms.untracked.InputTag("hpsPFTauDiscriminationByTightMuonRejection"),
   PFTauVLooseAntiElectron         = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA5VLooseElectronRejection"),
   PFTauLooseAntiElectron          = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA5LooseElectronRejection"),
   PFTauTightAntiElectron          = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA5TightElectronRejection")
)

