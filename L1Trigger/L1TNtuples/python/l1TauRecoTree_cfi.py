import FWCore.ParameterSet.Config as cms

l1TauRecoTree = cms.EDAnalyzer("L1TauRecoTreeProducer",
   period                          = cms.string("2016"),
   maxTau                          = cms.uint32(20),
   PFTauTag                        = cms.untracked.InputTag("hpsPFTauProducer"),
   PFTauDMFindingOld               = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs"),
   PFTauDMFinding                  = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
   PFTauTightIsoTag                = cms.untracked.InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
   PFTauLooseIsoTag                = cms.untracked.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
   PFTauLooseAntiMuon              = cms.untracked.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
   PFTauTightAntiMuon              = cms.untracked.InputTag("hpsPFTauDiscriminationByTightMuonRejection3"),
   PFTauVLooseAntiElectron         = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA6VLooseElectronRejection"),
   PFTauLooseAntiElectron          = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),
   PFTauTightAntiElectron          = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA6TightElectronRejection")
)

