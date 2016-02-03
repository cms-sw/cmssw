import FWCore.ParameterSet.Config as cms

l1MuonRecoTree = cms.EDAnalyzer("L1Muon2RecoTreeProducer",
   maxMuon                          = cms.uint32(20),
   MuonTag                          = cms.untracked.InputTag("muons"),
  #---------------------------------------------------------------------
  # TRIGGER MATCHING CONFIGURATION
  #---------------------------------------------------------------------
  # flag to turn trigger matching on / off
  triggerMatching = cms.untracked.bool(True),
  # maximum delta R between trigger object and muon
  triggerMaxDeltaR = cms.double(0.1),
  # trigger to match to, may use regexp wildcard as supported by ROOT's 
  # TString; up to now the first found match (per run) is used.
  isoTriggerNames = cms.vstring(
        "HLT_IsoMu18_v*",
        "HLT_IsoMu22_v*"
        #"HLT_IsoMu24_eta2p1_v*",
        #"HLT_IsoMu24_v*"
        ),
  triggerNames = cms.vstring(
        "HLT_Mu30_v*",
        "HLT_Mu40_v*"
        ),

  # data best guess: change for MC!
  triggerResults      = cms.InputTag("TriggerResults", "", "HLT"),
  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
  # name of the hlt process (same as above):
  triggerProcessLabel = cms.untracked.string("HLT"),
)

