import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource"
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histos.root")
)

process.dqmInfoJetMET = cms.EDAnalyzer("DQMEventInfo",
                                         subSystemFolder = cms.untracked.string('JetMET')
                                     )

process.load("DQMOffline.JetMET.dataCertificationJetMET_cfi")
process.dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string("/uscms_data/d1/hatake/DQM-data/DQM_V0001_R000066594__Cosmics__Commissioning08-PromptReco-v2__RECO.root"),
                              refFileName    = cms.untracked.string("/uscms_data/d1/hatake/DQM-data/DQM_V0001_R000066714__Cosmics__Commissioning08-PromptReco-v2__RECO.root"),
                              OutputFile     = cms.untracked.bool(False),
                              OutputFileName = cms.untracked.string("DQMResult.root"),
                              Verbose        = cms.untracked.int32(0),
                              TestType       = cms.untracked.int32(0)
)


process.p = cms.Path(process.dqmInfoJetMET*process.dataCertificationJetMET)
