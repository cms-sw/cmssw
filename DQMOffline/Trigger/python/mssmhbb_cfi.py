import FWCore.ParameterSet.Config as cms

# physics path monitor
mssmHbbPhysicsMonitor = cms.EDAnalyzer(
    "MssmHbbMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/Physics/"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets100_DoubleBTagCSV_0p92_DoublePFJets100MaxDeta1p6_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.5),
    muonPtMin = cms.double(5),
    muonEtaMax = cms.double(2.5),
    jetMuonDRMax = cms.double(0.4),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    muons = cms.InputTag("muons"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
       ),
       jetEta = cms.PSet (
          nbins = cms.int32 (10),
          xmin  = cms.double(-2.5),
          xmax  = cms.double(2.5),
       ),
       jetPhi = cms.PSet (
          nbins = cms.int32 (14),
          xmin  = cms.double(-3.5),
          xmax  = cms.double(3.5),
       ),
       jetBtag = cms.PSet (
          nbins = cms.int32 (20),
          xmin  = cms.double(0.8),
          xmax  = cms.double(1),
       ),
       jetsDR = cms.PSet (
          nbins = cms.int32 (40),
          xmin  = cms.double(0),
          xmax  = cms.double(4),
       ),
       muonPt = cms.PSet (
          nbins = cms.int32 (30),
          xmin  = cms.double(0),
          xmax  = cms.double(30),
       ),
       muonEta = cms.PSet (
          nbins = cms.int32 (10),
          xmin  = cms.double(-2.5),
          xmax  = cms.double(2.5),
       ),
       muonPhi = cms.PSet (
          nbins = cms.int32 (14),
          xmin  = cms.double(-3.5),
          xmax  = cms.double(3.5),
       ),
    ),
)

# online btagging monitor
mssmHbbBtagTriggerMonitor = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_SingleBTagCSV_0p92_DoublePFJets40_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (12),
          xmin  = cms.double(40),
          xmax  = cms.double(100),
       ),
       jetEta = cms.PSet (
          nbins = cms.int32 (10),
          xmin  = cms.double(-2.5),
          xmax  = cms.double(2.5),
       ),
       jetPhi = cms.PSet (
          nbins = cms.int32 (14),
          xmin  = cms.double(-3.5),
          xmax  = cms.double(3.5),
       ),
       jetBtag = cms.PSet (
          nbins = cms.int32 (20),
          xmin  = cms.double(0.8),
          xmax  = cms.double(1),
       ),
    ),
)
