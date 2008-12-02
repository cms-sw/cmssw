import FWCore.ParameterSet.Config as cms

dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string(""),
                              refFileName    = cms.untracked.string(""),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMResult.root"),
                              Verbose        = cms.untracked.int32(0),
                              TestType       = cms.untracked.int32(0)
)

