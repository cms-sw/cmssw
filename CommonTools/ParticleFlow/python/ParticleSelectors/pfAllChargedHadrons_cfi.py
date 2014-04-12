import FWCore.ParameterSet.Config as cms

pfAllChargedHadrons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoPileUpIso"),
    pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212),
    makeClones = cms.bool(True)
)



