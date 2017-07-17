import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer("PATMuonSlimmer",
    src = cms.InputTag("selectedPatMuons"),
    linkToPackedPFCandidates = cms.bool(True),
    pfCandidates = cms.VInputTag(cms.InputTag("particleFlow")),
    packedPFCandidates = cms.VInputTag(cms.InputTag("packedPFCandidates")), 
    saveTeVMuons = cms.string("pt > 100"), # you can put a cut to slim selectively, e.g. pt > 10
    dropDirectionalIso = cms.string("0"),
    dropPfP4 = cms.string("1"),
    slimCaloVars = cms.string("1"),
    slimKinkVars = cms.string("1"),
    slimCaloMETCorr = cms.string("1"),
    slimMatches = cms.string("1"),
    segmentsMuonSelection = cms.string("pt > 50"), #segments are needed for EXO analysis looking at TOF and for very high pt from e.g. Z' 
    saveSegments = cms.bool(True),
    modifyMuons = cms.bool(True),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

