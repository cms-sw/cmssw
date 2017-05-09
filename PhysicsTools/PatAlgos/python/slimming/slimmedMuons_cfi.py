import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer("PATMuonSlimmer",
    src = cms.InputTag("selectedPatMuons"),
    linkToPackedPFCandidates = cms.bool(True),
    pfCandidates = cms.VInputTag(cms.InputTag("particleFlow")),
    packedPFCandidates = cms.VInputTag(cms.InputTag("packedPFCandidates")), 
    saveTeVMuons = cms.string("pt > 100"), # you can put a cut to slim selectively, e.g. pt > 10
    modifyMuons = cms.bool(True),

    # Compute and store Mini-Isolation.
    # Implemention and a description of parameters can be found in:
    # PhysicsTools/PatUtils/src/PFIsolation.cc
    computeMiniIso = cms.bool(True),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParams = cms.vdouble(0.05, 0.2, 10.0, 0.5, 0.0001, 0.01, 0.01, 0.01, 0.0),

    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

