
import FWCore.ParameterSet.Config as cms

jetIDFailure = cms.EDFilter("JetIDFailureFilter",
        JetSource = cms.InputTag('patJetsPFlow'),
        MinJetPt      = cms.double(0.0),
        MaxJetEta     = cms.double(999.0),
        MaxNeutralHadFrac = cms.double(0.90),
        MaxNeutralEMFrac  = cms.double(0.95),
        debug         = cms.bool(False),
        taggingMode   = cms.bool(False),
)
