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
        "HLT_IsoMu18_v1",
        "HLT_IsoMu18_v2",
        "HLT_IsoMu18_v3",

        "HLT_IsoMu20_v1",
        "HLT_IsoMu20_v2",
        "HLT_IsoMu20_v3",
        "HLT_IsoMu20_v4",

        "HLT_IsoMu22_v1",
        "HLT_IsoMu22_v2",
        "HLT_IsoMu22_v3",

        "HLT_IsoMu24_v1",
        "HLT_IsoMu24_v2",

        "HLT_IsoMu27_v1",
        "HLT_IsoMu27_v2",
        "HLT_IsoMu27_v3",
        "HLT_IsoMu27_v4",
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

