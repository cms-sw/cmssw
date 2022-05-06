import FWCore.ParameterSet.Config as cms

slimmedDisplacedMuons = cms.EDProducer("PATMuonSlimmer",
    src = cms.InputTag("selectedPatDisplacedMuons"),
    linkToPackedPFCandidates = cms.bool(False),
    linkToLostTrack = cms.bool(False),
    pfCandidates = cms.VInputTag(cms.InputTag("particleFlow")),
    packedPFCandidates = cms.VInputTag(cms.InputTag("packedPFCandidates")), 
    lostTracks = cms.InputTag("lostTracks"),
    saveTeVMuons = cms.string("0"), # you can put a cut to slim selectively, e.g. pt > 10
    dropDirectionalIso = cms.string("0"),
    dropPfP4 = cms.string("1"),
    slimCaloVars = cms.string("1"),
    slimKinkVars = cms.string("1"),
    slimCaloMETCorr = cms.string("1"),
    slimMatches = cms.string("1"),
    segmentsMuonSelection = cms.string("pt > 50"), #segments are needed for EXO analysis looking at TOF and for very high pt from e.g. Z' 
    saveSegments = cms.bool(True),
    modifyMuons = cms.bool(True),
    modifierConfig = cms.PSet( modifications = cms.VPSet() ),
    trackExtraAssocs = cms.VInputTag(["displacedMuonReducedTrackExtras", "slimmedDisplacedMuonTrackExtras"]), 
)

