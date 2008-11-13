import FWCore.ParameterSet.Config as cms

dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string("/uscms/home/chlebana/DQM_V0001_R000063463__BeamHalo__BeamCommissioning08-PromptReco-v1__RECO.root"),
                              refFileName    = cms.untracked.string("/uscms/home/chlebana/DQM_V0001_R000063463__BeamHalo__BeamCommissioning08-PromptReco-v1__RECO.root"),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMResult.root"),
                              Verbose        = cms.untracked.int32(0),
                              TestType       = cms.untracked.int32(0)
)

