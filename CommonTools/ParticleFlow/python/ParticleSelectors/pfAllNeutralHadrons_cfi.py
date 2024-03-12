import FWCore.ParameterSet.Config as cms

pfAllNeutralHadrons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoPileUpIso"),
    pdgId = cms.vint32(111,130,310,2112),
    makeClones = cms.bool(True)
)



# foo bar baz
# yJ6m3T67M7roG
# IiP8L9v7f7E40
