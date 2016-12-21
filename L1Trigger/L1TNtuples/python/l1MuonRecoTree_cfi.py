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
        "HLT_IsoMu20_v*",
        "HLT_IsoMu22_v*",
        "HLT_IsoMu24_v*",
        "HLT_IsoMu27_v*",
        ),
  triggerNames = cms.vstring(
        "HLT_Mu30_v*",
        "HLT_Mu40_v*",
        "HLT_Mu50_v*",
        "HLT_Mu55_v*",
        # pA triggers
        "HLT_PAL3Mu12_v*",
        "HLT_PAL3Mu15_v*",
        "HLT_PAL2Mu12_v*",
        "HLT_PAL2Mu15_v*",
        ),

  # data best guess: change for MC!
  triggerResults      = cms.InputTag("TriggerResults", "", "HLT"),
  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
  # name of the hlt process (same as above):
  triggerProcessLabel = cms.untracked.string("HLT"),

  # muon track extrapolation to 1st station
  muProp1st = cms.PSet(
        useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
        useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra
        useSimpleGeometry = cms.bool(True),
        useStation2 = cms.bool(False),
  ),
  # muon track extrapolation to 2nd station
  muProp2nd = cms.PSet(
        useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
        useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra
        useSimpleGeometry = cms.bool(True),
        useStation2 = cms.bool(True),
        fallbackToME1 = cms.bool(False),
  ),
)

