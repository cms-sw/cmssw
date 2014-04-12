import FWCore.ParameterSet.Config as cms

pfAllMuons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoPileUp"),
    pdgId = cms.vint32( -13, 13),
    makeClones = cms.bool(True)
)

pfAllMuonsClones = cms.EDProducer("PFCandidateProductFromFwdPtrProducer",
                                  src = cms.InputTag("pfAllMuons")
                                  )



