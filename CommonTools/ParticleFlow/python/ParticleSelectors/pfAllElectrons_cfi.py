import FWCore.ParameterSet.Config as cms

pfAllElectrons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoMuon"),
    pdgId = cms.vint32(11,-11),
    makeClones = cms.bool(True)
)


pfAllElectronsClones = cms.EDProducer("PFCandidateProductFromFwdPtrProducer",
                                  src = cms.InputTag("pfAllElectrons")
                                  )





# foo bar baz
# Xx53IpNIkB38V
# oLgauVpg9LruC
