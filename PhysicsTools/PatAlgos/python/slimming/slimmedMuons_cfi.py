import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer("PATMuonSlimmer",
    src = cms.InputTag("selectedPatMuons"),
    linkToPackedPFCandidates = cms.bool(True),
    pfCandidates = cms.VInputTag(cms.InputTag("particleFlow")),
    packedPFCandidates = cms.VInputTag(cms.InputTag("packedPFCandidates")), 
    saveTeVMuons = cms.string("pt > 100"), # you can put a cut to slim selectively, e.g. pt > 10
    modifyMuons = cms.bool(True),
    computeMiniIso = cms.bool(True),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

