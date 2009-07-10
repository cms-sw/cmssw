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

process.dqmInfoJetMET = cms.EDFilter("DQMEventInfo",
                                         subSystemFolder = cms.untracked.string('JetMET')
                                     )

process.load("DQMOffline.JetMET.dataCertificationJetMET_cfi")
process.dataCertificationJetMET = cms.EDAnalyzer('DataCertificationJetMET',
                              fileName       = cms.untracked.string("jetMETMonitoring_cruzet98154.root"),
                              refFileName    = cms.untracked.string("jetMETMonitoring_cruzet98154.root"),
                              OutputFile     = cms.untracked.bool(True),
                              OutputFileName = cms.untracked.string("DQMCertResult_cruzet98154.root"),
                              Verbose        = cms.untracked.int32(1),
                              TestType       = cms.untracked.int32(0)
)

process.p = cms.Path(process.dqmInfoJetMET*process.dataCertificationJetMET)
