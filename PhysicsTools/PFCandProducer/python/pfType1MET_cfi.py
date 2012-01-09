import FWCore.ParameterSet.Config as cms

# Type-1 met corrections (AK5PFJets)
# remember about including ES producer definition e.g. JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff

pfType1MET = cms.EDProducer("Type1PFMET",
    inputUncorJetsTag = cms.InputTag("ak5PFJets"),
    jetEMfracLimit = cms.double(0.95), # to remove electron which give rise to jets
    jetMufracLimit = cms.double(0.95), # to remove electron which give rise to jets
    metType = cms.string("PFMET"), 
    jetPTthreshold = cms.double(20.0),
    # pfMET should be not corrected for HF 0.7
    inputUncorMetLabel = cms.string("pfMET"),
    corrector = cms.string("L2L3JetCorrectorAK5PF")
)

